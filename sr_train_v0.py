import  torch, os
import  numpy as np
import  scipy.stats
import matplotlib
matplotlib.use('Agg') 

from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta
from dataloader import dataloader as dl
import utility as util

from skimage.io import imsave, imread

import errors

import time



def mean_confidence_interval(accs, confidence=0.95):

    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)

    return m, h

def prepare(l, volatile=False): 
    device = torch.device('cuda')
    def _prepare(tensor):
        #if self.args.precision == 'half': tensor = tensor.half()
        return tensor.to(device, dtype=torch.float)
    return [_prepare(_l) for _l in l]

def main():
    ck = util.checkpoint(args)
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    ck.write_log(str(args))
    #t = str(int(time.time()))
    #t = args.save_name
    #os.mkdir('./{}'.format(t))
    # (ch_out, ch_in, k, k, stride, padding)
    config = [
        ('conv2d', [32, 16, 3, 3, 1, 1]),
        ('relu', [True]),              
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('conv2d', [3, 32, 3, 3, 1, 1]) 
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    # (Dataset) calculate the number of trainable tensors
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    ck.write_log(str(maml))
    ck.write_log('Total trainable tensors: {}'.format(num))

    # (Dataset) batchsz here means total episode number
    DL_MSI = dl.StereoMSIDatasetLoader(args)
    db = DL_MSI.train_loader
    dv = DL_MSI.valid_loader

    psnr = []
    l1_loss = []
    psnr_valid = []
    for epoch, (spt_ms, spt_rgb, qry_ms, qry_rgb) in enumerate(db):

        if epoch // args.epoch: break
        spt_ms, spt_rgb, qry_ms, qry_rgb = (spt_ms.to(device),
                spt_rgb.to(device), qry_ms.to(device), qry_rgb.to(device))

        # optimization is carried out inside meta_learner class, maml.
        accs, train_loss = maml(spt_ms, spt_rgb, qry_ms, qry_rgb)
        
        if epoch % args.print_every == 0:
            log_epoch =  'epoch: {} \ttraining acc: {}'.format(epoch, accs)
            ck.write_log(log_epoch) 
            psnr.append(accs)
            l1_loss.append(train_loss)
            ck.plot_loss(psnr, l1_loss, epoch, args.print_every)
            if epoch % args.save_every == 0:
                with torch.no_grad():
                    ck.save(maml.net, maml.meta_optim, epoch)
                    eval_psnr = 0 # psnr loss
                    for idx, (valid_ms, valid_rgb) in enumerate(dv):
                        #print('idx', idx)
                        valid_ms, valid_rgb = prepare([valid_ms, valid_rgb])
                        sr_rgb = maml.net(valid_ms)
                        sr_rgb = torch.clamp(sr_rgb, 0, 1)
                        eval_psnr += errors.find_psnr(valid_rgb, sr_rgb)
                        ############## plot PSNR here you idiot! ###########
                    psnr_valid.append(eval_psnr/25)
                    ck.plot_psnr(psnr_valid, epoch, args.save_every)
                    ck.write_log('Max PSNR is: {}'.format(max(psnr_valid))) 
                    imsave('./{}/validation/img_{}.png'.format(
                        ck.dir, epoch), np.uint8(sr_rgb[0,:,:,:].permute(
                                1, 2, 0).cpu().detach().numpy() * 255))
    ck.done()



if __name__ == '__main__':

    argparser = argparse.ArgumentParser() 
    argparser.add_argument('--seed', type=int, default=1, help='random seed')
    argparser.add_argument('--k', type=int, default=0,
        help='kth fold, values are from 0 to 10')
    argparser.add_argument('--epoch', type=int, help='epoch number',
            default=60000)
    argparser.add_argument('--num_workers', type=int, help='number of workers',
            default=1)
    argparser.add_argument('--pin_memory', type=bool, help='k shot for support set',
            default=True)
    argparser.add_argument('--shuffle', type=bool, help='shuffle', default=True)
    argparser.add_argument('--batch_size', type=int, help='batch_size',
            default=1)
    argparser.add_argument('--batchsz', type=int, help='batchsz',
            default=100000)
    argparser.add_argument('--crop_size', type=int, help='imgsz', default=120)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    #argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--task_num', type=int,
            help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr',
            type=float, help='meta-level outer learning rate',
            default=1e-3)
    argparser.add_argument('--update_lr', type=float,
            help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int,
            help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int,
            help='update steps for finetunning', default=10)
    argparser.add_argument('--save_every', type=int,
            help='task-level inner update learning rate', default=20)
    argparser.add_argument('--print_every', type=int,
            help='task-level inner update learning rate', default=5)
    argparser.add_argument('--folder_name', type=str,
            help='folder name', default='')
    argparser.add_argument('--load', type=str,
            help='load a model to continue training or for testing',
            default='.')
    argparser.add_argument('--save_name', type=str,
            help='save_name', default='.')
    argparser.add_argument('--test_only', type=bool,
            help='test_only', default=False)
    argparser.add_argument('--root', type=str,
            help='data', default='/flush5/sho092/Robust_learning/')
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    argparser.add_argument('--data_dist_same', type=str2bool, nargs='?',
            help=r'data distribution of the learning and meta learning are '
                 r'the same ==> True', default=False, const=True)


    args = argparser.parse_args()
    print(args)
    main()
