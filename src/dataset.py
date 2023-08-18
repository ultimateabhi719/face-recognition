#!/usr/bin/env python3.8
# coding: utf-8

import os
from PIL import Image
import PIL.ImageOps    
from tqdm.auto import tqdm
tqdm.pandas()
from typing import Dict, Any
import hashlib
import json

import numpy as np
import pandas as pd

import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def select_pairs(data_df, len_factor, prob_same, save_path=None):

    if len_factor<1:
        # subsample by ids
        data_pairs = data_df[data_df.ids.isin(pd.Series(data_df.ids.unique()).sample(frac=len_factor))]
    else:
        # if len_factor>1 make sure each image is present at least once
        len_factor -= 1
        data_pairs = pd.concat([data_df,data_df.sample(frac = len_factor, replace=True if len_factor>1 else False)])

    data_pairs = data_pairs.reset_index(drop=True).rename(columns = {'img':'img0', 'ids':'id0'})

    data_pairs['same_id'] = np.random.choice(2,size=len(data_pairs),p=(prob_same,1-prob_same))

    def sample_from_datadf(_id, same_id):
        if same_id==0:
            df = data_df[data_df.ids==_id]
        else:
            df = data_df[data_df.ids!=_id]

        return df[['img','ids']].sample(n=1).values.flatten().tolist()

    data_pairs['img1'], data_pairs['id1'] = zip(*data_pairs.progress_apply(lambda row: sample_from_datadf(row['id0'],row['same_id']), axis=1))
    
    if save_path:
        data_pairs.to_csv(save_path, index=False)

    return data_pairs

class SiameseDataset(Dataset):
    image_mean = np.array([0.485, 0.456, 0.406 ])
    image_std = np.array([0.229, 0.224, 0.225])

    inv_normalize = transforms.Normalize(
            mean=-image_mean/image_std,
            std=1/image_std
        )

    def __init__(self,datapath,transform=None,should_invert=False,len_factor=2, prob_same = 0.5):
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(
                                                mean=self.image_mean,
                                                std=self.image_std)]) 
        else:
            self.transform = transform
        self.should_invert = should_invert
        
        if type(datapath) == str:
            images, ids = list(zip(*dset.ImageFolder(root=datapath).imgs))        
            self.data_df = pd.DataFrame({'img':images, 'ids':ids})
        elif type(datapath) == tuple:
            root_dir, id_file = datapath
            self.data_df = pd.read_csv(id_file)
            self.data_df['img'] = self.data_df['img'].apply(lambda x:os.path.join(root_dir,x))
        else:
            raise ValueError("invalid datapath:", datapath)

        data_params = {'datapath':datapath, 'multfactor':len_factor, 'prob_same':prob_same}
        eda_path = f'eda/{dict_hash(data_params)}'
        datapairs_file = f"{eda_path}/pairs.csv"

        if not os.path.exists(eda_path):
            os.makedirs(eda_path)
            json.dump(data_params, open(f"{eda_path}/data_params.json", "w"))

        if not os.path.exists(datapairs_file):
            select_pairs(self.data_df, len_factor, prob_same, save_path=datapairs_file)

        self.data = pd.read_csv(datapairs_file)

    def __getitem__(self,index):
        img0 = Image.open(self.data.iloc[index]['img0']).convert("RGB")
        img1 = Image.open(self.data.iloc[index]['img1']).convert("RGB")
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return (self.data.iloc[index].img0,img0), (self.data.iloc[index].img1,img1) , torch.from_numpy(np.array(self.data.iloc[index].same_id,dtype=np.int_))
    
    def __len__(self):
        return len(self.data)


    


