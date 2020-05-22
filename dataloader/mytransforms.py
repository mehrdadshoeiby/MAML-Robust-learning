"""
define transforms I use for preparing my PIRM2018 training data
The input data is a dictionalary contraining lr: 'im_ms' and hr: 'im_rgb'
"""
import torch
import numpy as np
import torchvision.transforms as transforms

import random
import numbers 
from skimage.transform import rotate


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int,
        square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        im_ms, im_rgb = sample['im_ms'], sample['im_rgb']

        h, w = im_ms.shape[:2]
        new_h, new_w = self.output_size
        # the crop window needs to be twice as large for im_h 
        # need to edit here
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        im_ms = im_ms[top: top + new_h,
                      left: left + new_w]
        # the crop window needs to be twice as large for im_h
        im_rgb = im_rgb[top: top + new_h,
                      left: left + new_w]               

        return {'im_ms': im_ms, 'im_rgb': im_rgb}

    def __repr__(self):
        return self.__class__.__name__ + ': lr output size={}'.format(self.output_size)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im_ms, im_rgb = sample['im_ms'], sample['im_rgb']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im_ms = im_ms.transpose((2, 0, 1))
        im_rgb = im_rgb.transpose((2, 0, 1))        
        return {'im_ms': torch.from_numpy(im_ms).float(),
                'im_rgb': torch.from_numpy(im_rgb).float()}
    def __repr__(self):
        return self.__class__.__name__ 


class RandomRotate(object):
    """
    degrees is range of degrees to select from
    """
    def __init__(self, degrees, angle_list=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(r"If degrees is a single number",
                                 r"it must be positive.")
            self.degrees = (-degrees, degrees)
        self.angle_list = angle_list
        if self.angle_list==False:
            if len(degrees) != 2:
                raise ValueError(r"if degrees is a range of angles," 
                                 r"it must have a legth of 2")
        if len(degrees) > 2:
            self.degrees = degrees
            
    def __call__(self, sample):
        """
         sample (dictionary): lr and hr images to be rotated.
         lr and hr should be python images and not tensors
        
        Returns:
        dictionary containing lr and hr: Rotated image.
        """
        if self.angle_list:
            angle = random.choice(self.degrees)
        else:
            angle = random.uniform(self.degrees[0], self.degrees[1])
        im_ms, im_rgb = sample['im_ms'], sample['im_rgb']
        im_ms = rotate(im_ms, angle)
        im_rgb = rotate(im_rgb, angle)
        
        return {'im_ms': im_ms, 'im_rgb': im_rgb}

    def __repr__(self):

        format_string = self.__class__.__name__ + ': degrees={0}'.format(
                                                            self.degrees)
        return format_string

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample(dic contraining im_ms and im_rgb (not tensor)):
            Image to be flipped.

        Returns:
            dic containing flipped images.
        """
        im_ms, im_rgb = sample['im_ms'], sample['im_rgb']
        if random.random() < self.p:
            return {'im_ms': np.fliplr(im_ms).copy(),
                    'im_rgb': np.fliplr(im_rgb).copy()}
        return {'im_ms': im_ms, 'im_rgb': im_rgb}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize01(object):
    def __init__(self):
        # in case I want to use these to Normalise
        self.mean = [6339.81, 6491.63, 6665.94,8335.35, 10304.39, 6248.63,
                7052.06, 12352.15, 12790.39, 10291.37, 12096.63, 14150.68,
                13525.28, 12324.38]

        self.std = [5449.40, 5665.71, 5927.33, 7100.54, 8729.27,  5518.00,
               6127.89, 10379.74, 10699.50, 8691.75, 10093.53, 12148.49,
               11808.17, 10478.51]
    def __call__(self, sample):
        im_ms, im_rgb = sample['im_ms'], sample['im_rgb']
        im_ms = np.clip(im_ms/65535.0, 0, 1)
        im_rgb = im_rgb/255.0
#        Normal = transforms.Normalize(self.mean, self.std)
#        im_ms = Normal(im_ms)
        return {'im_ms': im_ms, 'im_rgb': im_rgb}
    def __repr__(self):
        return self.__class__.__name__ + ': Normalise images between 0 and 1'
