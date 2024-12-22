import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_data_loader(
    data_root, 
    mode, 
    batch_size, 
    image_size=(320,320), 
    num_workers=0, 
    augment=False,
):
    """
    Get the data loader for the specified dataset and mode.
    :param data_root: the root directory of the whole dataset 根目录
    :param mode: the mode of the dataset, which can be 'train', 'val', or 'test' 
    :param image_size: the target image size for resizing 
    :param batch_size: the batch size 
    :param num_workers: the number of workers for loading data in multiple processes
    :param augment: whether to use data augmentation 是否数据增强
    :return: a data loader
    """

    data_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if mode == "train" and augment:        
        #data_transforms.append(transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.1))
        data_transforms.append(transforms.RandomPerspective(distortion_scale=0.1,p=0.1))
        data_transforms.append(transforms.RandomRotation(10))        
        data_transforms.append(transforms.RandomGrayscale(p=0.1))

    data_transforms = transforms.Compose(data_transforms)

    dataset = ImageFolder(os.path.join(data_root,mode),data_transforms,target_transform=None,)
    loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=(mode=='train'),num_workers=num_workers)


    return loader
