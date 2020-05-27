import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import glob

import numpy as np
import collections
from PIL import Image
from skimage.io import imread
import csv
import random

#from dataloader import mytransforms

class StereoMSITrain(Dataset):
    """
    put mini-imagenet files as:
    root:
        |- images/*.npy-*.tiff includes all images
    """

    def __init__(self, args, root, mode, setsz, querysz, batchsz,
                 mytransforms, k):
        """
        :param root: root self.root of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param setsz: actual batch size for each task 
        :param k_query: actual batch size for validation
        :param k: the fold number
        """
        self.root = root
        self.mytransform = mytransforms
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.task_num = args.task_num
        self.crop_size = args.crop_size
        # self.setsz = self.n_way * self.k_shot # num of samples per set
        self.setsz = setsz
        self.querysz = querysz
        self.data_dist_same = args.data_dist_same
        train_list = open(self.root + 'train_k{}.txt'.format(k), 'r')
        temp = list(train_list)
        self.train_list = [i[:-1] for i in temp]
        print('shuffle DB :%s, b:%d,  %d-setsz, %d-querysz' % ( mode, batchsz,
            setsz, querysz))
                
        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return: indices of support and query batch
        """
        # lists to store training and validation batches (ms) 
        # I need two more to store rgb images.
        # support/training batch/episode of 5 sets
        self.support_batch = [[] for i in range(batchsz)]
        # query/valid/test batch/episode of 5 sets
        self.query_batch = [[] for i in range(batchsz)]
  
        if self.data_dist_same==True:
            img_idx1 = np.arange(0, 250)
            #img_idx2 = np.arange(201, 251)
        elif self.data_dist_same==False and self.data_dist_shuffle==False:
            img_idx1 = np.arange(0, 200)
            img_idx2 = np.arange(200, 250)
        else:
            img_idx = np.arange(0,250)

        for b in range(batchsz):
            if self.data_dist_same==True and self.data_dist_shuffle==False:
                random.shuffle(img_idx1)
            elif self.data_dist_same==False and self.data_dist_shuffle==False:
                random.shuffle(img_idx1)
                random.shuffle(img_idx2)
            else: 
                random.shuffle(img_idx)
                img_idx1 = imd_idx[0:200]
                img_idx2 = img_idx[200:250]
                

            for episode in range(self.task_num): 
                # When DataSet class is initialised, create_batch is run once
                # for each batch
                # 1. shuffle 
                # 2. select 5 sets of 50 training (40 support + 10 query images
                # out of 250 images. validation and testing won't change.
                # batchsz: total number of possible batches (a large number)
                # to cover the possible data distribution.
                if self.data_dist_same==True:
                    self.support_batch[b].append(
                        img_idx1[episode*50:(episode+1)*50-10])
                    self.query_batch[b].append(
                        img_idx1[episode*50+40:(episode+1)*50])
                else:
                # learning and meta-learning from different data distribution
                    self.support_batch[b].append(img_idx1[episode*40:(episode+1)*40])
                    self.query_batch[b].append(img_idx2[episode*10:(episode+1)*10])
                # learning and meta-leaning from the same data distribution
 

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return: a set
        loops over a list of paths and reads the images and stores them in an
            array
        """
        # create tensor to store images for 1 meta-learning epoch! that is,
        # task_num episodes, each containing setsz for training and querysz for
        # validation.
        # [task_num, setsz/querysz, 16, crop_size, crop_size]
        support_ms = torch.FloatTensor(self.task_num, self.setsz, 16,
                self.crop_size, self.crop_size)
        # [querysz, 16, resize, resize]
        query_ms = torch.FloatTensor(self.task_num,self.querysz, 16,
                self.crop_size, self.crop_size)
        # [setsz, 3, resize, resize]
        support_rgb = torch.FloatTensor(self.task_num, self.setsz, 3,
                self.crop_size, self.crop_size)
        # [querysz, 3, resize, resize]
        query_rgb = torch.FloatTensor(self.task_num, self.querysz, 3,
                self.crop_size, self.crop_size)

        # create a list of self.root to images by randomly selecting one of the
        # batchsz metabatches and convert it to a 2D list (episode, path)
        # Path does not contain extension.
        # I dont need two lists for rgb and ms (4 in total)  I only need two
        # lists flatten_support and flatten_query because rgb and ms have the
        # same names only different extensions.
#############################################################################
####Note: flatten_support must contain self.root to file without the extension ###
#############################################################################

        flatten_support = [[] for i in range(self.task_num)]
        for episode in range(5):
            flatten_support[episode] = [item for item in
                    self.support_batch[index][episode]]
#        flatten_support_rgb = [[] for i in range(self.task_num)]
#        for episode in range(5):
#            flatten_support_rgb[episode] = [item for item in
#                    self.support_batch[index][episode]] 

        flatten_query = [[] for i in range(self.task_num)]
        for episode in range(5):
            flatten_query[episode] = [item for item in
                    self.query_batch[index][episode]]
#        flatten_query_rgb = [[] for i in range(self.task_num)]
#        for episode in range(5):
#            flatten_query_rgb[episode] = [item for item in
#                    self.support_batch[index][episode]]          
        # reads all the images and stores them in a dictionary
     
        for i, episode in enumerate(flatten_support):
            j = 0
            for item in episode: # here I'll add image name to param: root 
                # here, item should be the item_th image is a list that I would
                # have generated earlier. In __init__ I can convert the .txt
                # attributed to a kth fold to a python list and item is the
                # item_th element of the list.
                #print('item for support is :', item)
                #print('length of train_list is: ', len(self.train_list))
                n = self.train_list[item]
                im_ms = self.root + 'source/train/{}.npy'.format(n)
                im_rgb = self.root +'source/train/{}.tiff'.format(n)

                im_ms = np.array(np.load(im_ms), dtype=np.float)
                im_rgb = np.array(imread(im_rgb), dtype=np.float)
                sample_support = {'im_ms': im_ms, 'im_rgb': im_rgb}
                #print(sample_support)
                if self.mytransform:
                    img_dict = self.mytransform(sample_support)
                    #print(img_dict['im_ms'].shape)
                    #print(support_ms.shape)
                    #print('i',i,'j', j)
                    support_ms[i, j] = img_dict['im_ms']
                    support_rgb[i, j] = img_dict['im_rgb']
                    j += 1 

        for i, episode in enumerate(flatten_query):
            j = 0
            for item in episode: # here I'll add image name to param: path
                # modify as above (~20 lines)
                #print('item for query is: ', item)
                #print('length of train list is: ', len(self.train_list))
                n = self.train_list[item]
                im_ms = self.root + 'source/train/{}.npy'.format(n)
                im_rgb = self.root +'source/train/{}.tiff'.format(n)
                
                im_ms = np.array(np.load(im_ms), dtype=np.float)
                im_rgb = np.array(imread(im_rgb), dtype=np.float)
                sample_query = {'im_ms': im_ms, 'im_rgb': im_rgb}

                if self.mytransform:
                    img_dict = self.mytransform(sample_query)
                    #print(query_ms.shape) 
                    #print('i',i,'j', j)
                    query_ms[i, j] = img_dict['im_ms']
                    query_rgb[i, j] = img_dict['im_rgb']
                    j += 1
        #support_ms = support_ms.squeeze()
        #support_rgb = support_rgb.squeeze()
        #query_ms = query_ms.squeeze()
        #query_rgb = query_rgb.squeeze()
    
        return support_ms, support_rgb, query_ms, query_rgb

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small
        # batch size of sets.

        return self.batchsz


class StereoMSIValidDataset(Dataset):
    """
    all the training data should be stored in the same folder
    format for lr image  ==> "image_{}_lr2".format(idx)
    """

    def __init__(self, args, root, mytransform, k):
        self.root = root
        self.mytransform = mytransform
        valid_list = open(self.root + 'valid_k{}.txt'.format(k), 'r')
        temp = list(valid_list)
        self.valid_list = [i[:-1] for i in temp]

    def __len__(self):
        # this should be the lenght of validation python list generated for a
        # .txt file in __initi__?
        return len(self.valid_list)

    def __getitem__(self, idx):
        # validation dataset idx starts from 251
        # select image path from validation list using idx ==> modify below
        n = self.valid_list[idx]
        img_ms = self.root + 'source/train/{}.npy'.format(n)
        img_rgb = self.root +'source/train/{}.tiff'.format(n)
        #print(im_ms)
        #print(im_rgb)
#        img_ms = os.path.join(self.root,
#                                "valid/{}.npy".format(str(idx+251)))
#        img_rgb = os.path.join(self.root,
#                                "valid/{}.tiff".format(str(idx+251)))
#
        img_ms = np.array(np.load(img_ms), dtype=np.float)
        img_rgb = np.array(imread(img_rgb), dtype=np.float)
        sample_valid = {'im_ms': img_ms, 'im_rgb': img_rgb}

        if self.mytransform:
            sample_valid = self.mytransform(sample_valid)
        # read validation/testing dataset
        valid_ms = sample_valid['im_ms']
        valid_rgb = sample_valid['im_rgb']

        return valid_ms, valid_rgb


#class StereoMSIValidDataset(Dataset):
#    """
#    all the training data should be stored in the same folder
#    format for lr image  ==> "image_{}_lr2".format(idx)
#    """
#
#    def __init__(self, args, root, mytransform):
#        self.root = root
#        self.mytransform = mytransform
# 
#    def __len__(self):
#        # find the number of labels hence lenght of the dataset.
#        return len(glob.glob1(os.path.join(self.root + 'valid'),
#                              '*.tiff'))
#
#    def __getitem__(self, idx):
#        # validation dataset idx starts from 201
#        img_ms = os.path.join(self.root,
#                                "valid/{}.npy".format(str(idx+251)))
#        img_rgb = os.path.join(self.root,
#                                "valid/{}.tiff".format(str(idx+251)))
#
#        img_ms = np.array(np.load(img_ms), dtype=np.float)
#        img_rgb = np.array(imread(img_rgb), dtype=np.float)
#        sample_valid = {'im_ms': img_ms, 'im_rgb': img_rgb}
#
#        if self.mytransform:
#            sample_valid = self.mytransform(sample_valid)
#        # read validation/testing dataset
#        valid_ms = sample_valid['im_ms']
#        valid_rgb = sample_valid['im_rgb']
#
#        return valid_ms, valid_rgb
#

if __name__ == '__main__':

    
    # the following episode is to view one set of images via tensorboard.
    # from torchvision.utils import make_grid
    # from matplotlib import pyplot as plt
    # from tensorboardX import SummaryWriter
    # import time
    # plt.ion()
    print()

    import numpy as np
    import torch
    from torch import nn
    from  torch.utils.data import DataLoader

    #from dataloader import StereoMSI_MAML as DataSet
    import dataloader as dl
    import argparse
    import matplotlib.pyplot as plt
    import pudb

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
    
    DL_MSI = dl.StereoMSIDatasetLoader(args)
    
    train_loader = DL_MSI.train_loader
     
    i, j = next(enumerate(train_loader))
    # pu.db
    task = 0
    image = 0

    j1_rgb = j[1].squeeze()[task,image,:,:,:].permute((1,2,0)).numpy() 
    j0_ms = j[0].squeeze()[task,image,:,:,:].permute((1,2,0)).numpy() 
    print(j1_rgb.shape)
    print(j0_ms.shape)
    plt.figure(), plt.imshow(np.average(j0_ms, axis=2))
    plt.figure(), plt.imshow(j1_rgb)
    plt.show()

