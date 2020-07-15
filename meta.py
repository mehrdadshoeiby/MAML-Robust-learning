import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy

import errors


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
            return param_group['lr']

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        # self.k_spt = args.k_spt # support: not used: when used k_spt + k_qry 
        # self.k_qry = args.k_qry # should be 50 if task_num is 5 (250 training
        self.task_num = args.task_num # images)
        self.update_step = args.update_step # task level update step.
        self.update_step_test = args.update_step_test
        # inner loop: the network, takes network structure and input size
        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.meta_optim,
                        milestones=[2000, 4000],
                        gamma=0.5)
        self.log = [] 
        print('init in Meta class running!')

    # I dont think this function has been used here!
    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, spt_ms, spt_rgb, qry_ms, qry_rgb, epoch):
        """
        :b:             number of tasks/batches.
        :setsz:         number of training pairs?
        :querysz        number of test pairs for few shot
        :param spt_ms:    [task_num, setsz, 16, h, w]
        :param spt_rgb:   [task_num, querysz, 3, h, w] 
        :param qry_ms:    [task_num, setsz, 16, h, w]
        :param qry_rgb:   [task_num, querysz, 3, h, w]

        :return:
        """

        spt_ms = spt_ms.squeeze()
        spt_rgb = spt_rgb.squeeze()
        qry_ms = qry_ms.squeeze()
        qry_rgb = qry_rgb.squeeze()

        task_num, setsz, c, h, w = spt_ms.size()
        _, querysz, c, _, _ = qry_ms.size()
        # losses_q[k] is the loss on step k of gradient descent (inner loop)
        losses_q = [0 for _ in range(self.update_step + 1)]
        # accuracy on step i of gradient descent (inner loop)
        corrects = [0 for _ in range(self.update_step + 1)]
        if (epoch < 4001):
            if (epoch%2000==0)  and (epoch > 1):
                decay = 2 #(epoch // 5) + 1
                self.update_lr = self.update_lr / decay
        print('outer loop lr is: ', self.update_lr) 
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0, k is update step
            logits = self.net(spt_ms[i], vars=None, bn_training=True)
            loss = F.smooth_l1_loss(logits, spt_rgb[i])
            # create a log with task_num x k
            #print(loss.item())
            # the sum of graidents of outputs w.r.t the input
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(
                            lambda p: p[1] - self.update_lr * p[0],
                            zip(grad, self.net.parameters())
                            ))
            # what are these two torch.no_grad()s about?????????????????????
            # the first one calculates accuracy right after initialization
            # which makes sense, the second one is doing an update...why?????
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(qry_ms[i], self.net.parameters(),
                        bn_training=True)
                loss_q = F.smooth_l1_loss(logits_q, qry_rgb[i])
                losses_q[0] += loss_q # adding loss?!

                pred_q = logits_q # logits_q used to be cross_entropy loss, and
                # go through softmax to become pred_q.
                # calculate PSNR
                correct = errors.find_psnr(pred_q, qry_rgb[i])
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(qry_ms[i], fast_weights, bn_training=True)
                loss_q = F.smooth_l1_loss(logits_q, qry_rgb[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = logits_q
                correct = errors.find_psnr(pred_q, qry_rgb[i])
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(spt_ms[i], fast_weights, bn_training=True)
                loss = F.smooth_l1_loss(logits, spt_rgb[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                    zip(grad, fast_weights)))

                logits_q = self.net(qry_ms[i], fast_weights, bn_training=True)
                self.valid_img = logits_q
                # loss_q will be overwritten and we just keep the loss_q on
                # last update step ==> losses_q[-1]
                loss_q = F.smooth_l1_loss(logits_q, qry_rgb[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = logits_q
                    # convert to numpy
                    correct = errors.find_psnr(pred_q, qry_rgb[i])  
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # self.log[-1] += loss.item()
        # optimize theta parameters
        # In the Learner the update is with respect to accuracy of the training
        # set, but for meta_learner the meta_update is with respect to the test
        # set of each episode.
        self.meta_optim.zero_grad()
        loss_q.backward() # backwards through grad above ==> d(loss_q)/d(grad)
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        accs = np.average(np.array(corrects[-1])) #/ (querysz * task_num)
        print('inner loop lr is: ', self.get_lr(self.meta_optim))
        return accs, loss_q 

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
                return param_group['lr']




















    # finetunning is for training the network in a new task (fine tuning it)
    def finetunning(self, spt_ms, spt_rgb, qry_ms, qry_rgb):
        """
        :param spt_ms:    [task_num, setsz, 16, h, w]
        :param spt_rgb:   [task_num, setsz, 16, h, w]
        :param qry_ms:    [task_num, setsz, 16, h, w]
        :param qry_rgb:   [task_num, setsz, 16, h, w]
        :return:
        """
        assert len(spt_ms.shape) == 4

        querysz = qry_ms.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(spt_ms)
        loss = F.cross_entropy(logits, spt_rgb)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
            zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            pred_q = net(qry_ms, net.parameters(), bn_training=True)
            # [setsz]
            # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, qry_rgb).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            pred_q = net(qry_ms, fast_weights, bn_training=True)
            # [setsz]
            # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, qry_rgb).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(spt_ms, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, spt_rgb)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                zip(grad, fast_weights)))

            pred_q = net(qry_ms, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.smooth_l1_loss(pred_q, qry_rgb)

            with torch.no_grad():
                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, qry_rgb).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        
        del net

        accs = np.array(corrects) / querysz

        return accs

def main():
    pass

if __name__ == '__main__':
    main()
