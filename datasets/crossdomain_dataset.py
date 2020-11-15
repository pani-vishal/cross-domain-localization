import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import sys

path_base = os.path.join(os.getcwd(), '..')

sys.path.insert(1, path_base)
from utils.utility import *

class CrossDomainDataset(Dataset):
    """Dataset of static and dynamic domain."""
    def __init__(self, dict_df, embeddings, dict_arr_imgs, dict_stats, static_tfm=None, dynamic_tfm=None, xtra_tfm=None):
        self.dict_arr_imgs = dict_arr_imgs
        self.dict_df = dict_df
        self.embeddings = embeddings
        self.static_tfm = static_tfm
        self.dynamic_tfm = dynamic_tfm
        self.dynamic_tfm_extra = xtra_tfm
        self.to_tensor = transforms.ToTensor()

        self.dict_stats = dict_stats
        self.dict_normalize = {}
        for k, v in dict_stats.items():
          self.dict_normalize[k] = transforms.Normalize(mean=v[0], std=v[1])
        
    def __len__(self):
        return len(self.dict_df["night"])
    
    def __getitem__(self, idx):
        """Returns a dict with entries of each domain"""
        dict_sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        for domain, df in self.dict_df.items():
            idx_static, idx_dynamic, label = (df.iloc[idx, 0],
                                              df.iloc[idx, 1],
                                              df.iloc[idx, 2])
            
            # load both images and apply tfms
            static = self.embeddings[idx_static]
            dynamic = self.dict_arr_imgs[domain][int(idx_dynamic)] 
        
            if self.dynamic_tfm:
                dynamic = self.dynamic_tfm(image=dynamic)["image"]
            if self.dynamic_tfm_extra:
                dynamic = self.dynamic_tfm_extra(dynamic)

            static =  torch.from_numpy(np.asarray(static)).float()
            dynamic = self.dict_normalize[domain](self.to_tensor(dynamic))
            label = torch.from_numpy(np.asarray(label)).long()
            dict_sample[domain] = [[static, dynamic], label]
        return dict_sample

    
    def get_img_pairs(self, idx):
        """To display the images, as __get__() returns the embeddings for the static domain"""
        dict_sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        for domain, df in self.dict_df.items():
            idx_static, idx_dynamic, label = (df.iloc[idx, 0],
                                              df.iloc[idx, 1],
                                              df.iloc[idx, 2])
            
            # load both images and apply tfms
            static = self.dict_arr_imgs["static"][int(idx_static)] 
            dynamic = self.dict_arr_imgs[domain][int(idx_dynamic)] 
            
            if self.static_tfm: 
                static = self.static_tfm(image=static)["image"]
            if self.dynamic_tfm:
                dynamic = self.dynamic_tfm(image=dynamic)["image"]
            if self.dynamic_tfm_extra:
                dynamic = self.dynamic_tfm_extra(dynamic)

            static = self.dict_normalize["static"](self.to_tensor(static))
            dynamic = self.dict_normalize[domain](self.to_tensor(dynamic))
            label = torch.from_numpy(np.asarray(label)).long()
            dict_sample[domain] = [[static, dynamic], label]
        return dict_sample

    def get_domain_stats(self, domain):
        """Return domain stats, useful when each domain treated with different stats"""
        return self.dict_stats["static"], self.dict_stats[domain]