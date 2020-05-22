import torch
from torch import nn
from  torch.utils.data import DataLoader

from dataloader import StereoMSI_MAML as DataSet
from dataloader import dataloader as dl
import argparse

import pudb

def main():

#    train_dataset = DataSet.StereoMSITrain(args,
#                root='/home/sho092/Documents/SSD_data/ms_rgb_data/',
#                mode='train', setsz=40, querysz=10, batchsz=10000)

    DL_MSI = dl.StereoMSIDatasetLoader(args)
    train_loader = DL_MSI.train_loader
    k = 0
    l = 0
    i,j = next(enumerate(train_loader))
    pu.db



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number',
            default=60000)
    argparser.add_argument('--num_workers', type=int, help='number of workers',
            default=1)
    argparser.add_argument('--pin_memory', type=bool, help='k shot for support set',
            default=True)
    argparser.add_argument('--shuffle', type=bool, help='shuffle', default=True)
    argparser.add_argument('--batch_size', type=int, help='batch_size',
            default=1)
    argparser.add_argument('--crop_size', type=int, help='imgsz', default=64)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int,
            help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr',
            type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float,
            help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int,
            help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int,
            help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    print(args)
    main()
