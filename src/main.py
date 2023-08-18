#!/usr/bin/env python3.8
# coding: utf-8

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import glob
from natsort import natsorted
import itertools
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from SiameseNetwork import ContrastiveLoss, SiameseNetwork
from dataset import SiameseDataset

def imshow(img,text=None,save_img=None):
    npimg = np.asarray(img)#.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    # npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()    
    if save_img:
        plt.savefig(save_img)

def train_epoch(train_dataloader, net, loss_fn, optimizer, epoch, save_freq, logwriter, savemodel_path, saveloss_path):
    net.train()
    loss_fn.train()

    epoch_loss = 0
    pbar = tqdm(train_dataloader, ncols=100, leave=False)
    pbar.set_description(f"Epoch {epoch}")
    for i, data in enumerate(pbar):
        (_,img0), (_,img1), label = data
        img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss = loss_fn(output1,output2,label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        logwriter.add_scalar('training loss', loss.item(), epoch * len(train_dataloader) + (i+1))
        if (i+1) % save_freq == 0 :
            torch.save(net.state_dict(), savemodel_path.format(epoch,i+1))
            torch.save(loss_fn.state_dict(), saveloss_path.format(epoch,i+1))

    torch.save(net.state_dict(), savemodel_path.format(epoch,'N'))
    torch.save(loss_fn.state_dict(), saveloss_path.format(epoch,'N'))
    epoch_loss = epoch_loss/len(train_dataloader)

    return epoch_loss

@torch.no_grad()
def evaluate_loss(val_dataloader, net, loss_fn, logwriter=None, x0=0, saveimg_folder = None):
    net.eval()
    loss_fn.eval()
    val_loss = 0
    pbar = tqdm(val_dataloader, ncols=100, leave=False)
    pbar.set_description("Val")
    for i, data in enumerate(pbar):
        (_,img0), (_,img1), label = data
        img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

        output1,output2 = net(img0,img1)
        loss = loss_fn(output1,output2,label)
        
        val_loss += loss.item()
        if logwriter:
            logwriter.add_scalar('val loss', loss.item(), x0+(i+1))

        if saveimg_folder:
            concatenated = torch.cat((img0,img1),0)        
            euclidean_distance = F.pairwise_distance(output1, output2)
            imshow(torchvision.utils.make_grid(SiameseDataset.inv_normalize(concatenated)).permute(1,2,0).cpu(), 
                                                             f'Label: {label[0].item()}; Dist: {euclidean_distance.item():.2f}',
                                                             save_img=os.path.join(saveimg_folder,f'{i}.png'))

    val_loss = val_loss/len(val_dataloader)

    return val_loss

def main(model_params, data_params, train_params, device):
    train_dataset = SiameseDataset(data_params['train_datapath'], len_factor=data_params['traindata_multfactor'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=False, num_workers=0, batch_size=train_params['batch_size'])

    val_dataset = SiameseDataset(data_params['val_datapath'], len_factor=data_params['valdata_multfactor'])
    val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=False, num_workers=0, batch_size=train_params['batch_size'])


    loss_fn = ContrastiveLoss().to(device)
    # loss_fn = LearnedLoss().to(device)
    net = SiameseNetwork(**model_params).to(device)
    if train_params['resume_dir']:
        resume_model = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_model'].format('*','*'))))[-1]
        resume_loss = re.sub(train_params['save_model'].format('(.*)','(.*)'), train_params['save_loss'].format('\\1','\\2'), resume_model)
        net.load_state_dict(torch.load(resume_model, map_location=device))
        loss_fn.load_state_dict(torch.load(resume_loss, map_location=device))

    optimizer = optim.Adam(itertools.chain.from_iterable([net.parameters(), loss_fn.parameters()]),lr = 0.0005 )
    logwriter = SummaryWriter(train_params['save_prefix'])
    for epoch in range(train_params['epochs']):
        epoch_loss = train_epoch(train_dataloader, 
                                 net, 
                                 loss_fn, 
                                 optimizer, 
                                 epoch, 
                                 train_params['save_freq'], 
                                 logwriter, 
                                 os.path.join(train_params['save_prefix'],train_params['save_model']),
                                 os.path.join(train_params['save_prefix'],train_params['save_loss'])
                                 )
        val_loss = evaluate_loss(val_dataloader, net, loss_fn, logwriter=logwriter, x0=epoch*len(val_dataloader))
        print(f"Epoch {epoch}:: train_loss : {epoch_loss:.2f}, val_loss : {val_loss:.2f}")


    logwriter.close()




    test_dataset = SiameseDataset(data_params['test_datapath'], len_factor=data_params['testdata_multfactor'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=False, num_workers=0, batch_size=1)

    test_folder = os.path.join(train_params['save_prefix'], 'test_images')
    os.makedirs(test_folder)
    test_loss = evaluate_loss(test_dataloader, net, loss_fn, logwriter=None, saveimg_folder = test_folder)



if __name__ == "__main__":

    model_params = {'fc_dim':256, 
                    'out_dim': 32}

    # data_params = {'train_datapath':"data/att_faces/Training",
    #                'traindata_multfactor':8,
    #                'val_datapath':"data/att_faces/Testing",
    #                'valdata_multfactor':4,
    #                'test_datapath':"data/att_faces/Testing",
    #                'testdata_multfactor':.2}

    data_params = {'train_datapath':('data/celebA/Img/img_align_celeba','data/celebA/identity_CelebA_filt.train.csv'),
                   'traindata_multfactor':0.6,
                   'val_datapath':('data/celebA/Img/img_align_celeba','data/celebA/identity_CelebA_filt.val.csv'),
                   'valdata_multfactor':1,
                   'test_datapath':('data/celebA/Img/img_align_celeba','data/celebA/identity_CelebA_filt.test.csv'),
                   'testdata_multfactor':1}


    train_params = {'batch_size':648,
                    'resume_dir':None,
                    'save_prefix':"runs/celeba_fc256_out32_log",
                    'epochs':20,
                    'save_freq':1000,

                    'save_model':"siamese_epoch{}_batch{}.pth",
                    'save_loss':"loss_epoch{}_batch{}.pth"
                   }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    os.makedirs(train_params['save_prefix'], exist_ok=True)
    torch.save([model_params, data_params, train_params],os.path.join(train_params['save_prefix'],'params.pth'))

    main(model_params, data_params, train_params, device)
