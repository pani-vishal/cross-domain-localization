import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import sys

path_base = os.path.join(os.getcwd(), '..')

sys.path.insert(1, path_base)
from utils.utility import *


class TripletDataset(Dataset):
    """Dataset of static and dynamic domain."""
    def __init__(self, data_arr, tfm=None, xtra_tfm=None):
        """
        Args:
            data_arr (np.array): Image array.
            static_tfm, dynamic_tfm (callable, optional): Optional transform to be applied
                on a static or dynamic sample.
            xtra_tfm (callable, optional): transform on all items, use this to convert
                to torch tensor
        """
        self.tfm = tfm
        self.data_arr = data_arr
        self.dynamic_tfm_extra = xtra_tfm
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                              std=(0.229, 0.224, 0.225))
        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor = idx
        positive = get_similar_image_no(anchor, len(self.data_arr))
        if random.random() > 0.5:
            negative = get_different_close_image_no(anchor, len(self.data_arr))
        else:
            negative = get_different_image_no(anchor, len(self.data_arr)) 

        # load both images and apply tfms
        anchor = self.data_arr[anchor] 
        positive = self.data_arr[positive]
        negative = self.data_arr[negative]

            
        if self.tfm:
            anchor = self.tfm(image=anchor)["image"]
            positive = self.tfm(image=positive)["image"]
            negative = self.tfm(image=negative)["image"]
        if self.dynamic_tfm_extra:
            anchor = self.dynamic_tfm_extra(anchor)
            positive = self.dynamic_tfm_extra(positive)
            negative = self.dynamic_tfm_extra(negative)

        anchor = self.normalize(self.to_tensor(anchor))
        positive = self.normalize(self.to_tensor(positive))
        negative = self.normalize(self.to_tensor(negative))

        return (anchor, positive, negative), []