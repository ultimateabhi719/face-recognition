#!/usr/bin/env python3.8
# coding: utf-8

import os
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
from natsort import natsorted
import itertools
from tqdm.auto import tqdm
import sklearn.metrics

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import TrialState

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


def compute_stats(y_true, probs):
    _, _, thresholds = sklearn.metrics.roc_curve(y_true, probs)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(sklearn.metrics.accuracy_score(y_true, [m > thresh for m in probs]))

    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max() 
    max_accuracy_threshold =  thresholds[accuracies.argmax()]

    roc_auc = sklearn.metrics.roc_auc_score(y_true, probs)

    return max_accuracy, max_accuracy_threshold, roc_auc

def train_epoch(train_dataloader, net, loss_fn, optimizer, device, epoch, save_freq, logwriter, savemodel_path, saveloss_path):
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
        if logwriter:
            logwriter.add_scalar('training loss', loss.item(), epoch * len(train_dataloader) + (i+1))
        if (i+1) % save_freq == 0 :
            torch.save(net.state_dict(), savemodel_path.format(epoch,i+1))
            torch.save(loss_fn.state_dict(), saveloss_path.format(epoch,i+1))

    torch.save(net.state_dict(), savemodel_path.format(epoch,'N'))
    torch.save(loss_fn.state_dict(), saveloss_path.format(epoch,'N'))
    epoch_loss = epoch_loss/len(train_dataloader)

    return epoch_loss

@torch.no_grad()
def evaluate_loss(dataloader, net, loss_fn, device, logwriter=None, x0=0, saveimg_folder = None):
    net.eval()
    loss_fn.eval()

    euc_dist_, label_ = [], []
    total_loss = 0
    pbar = tqdm(dataloader, ncols=100, leave=False)
    pbar.set_description("Val")
    for i, data in enumerate(pbar):
        (_,img0), (_,img1), label = data
        img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

        output1,output2 = net(img0,img1)
        loss = loss_fn(output1,output2,label)
        
        total_loss += loss.item()
        if logwriter:
            logwriter.add_scalar('val loss', loss.item(), x0+(i+1))

        euclidean_distance = F.pairwise_distance(output1, output2)
        euc_dist_.append(euclidean_distance.detach().clone().cpu().numpy())
        label_.append(label.detach().clone().cpu().numpy())

        if saveimg_folder:
            assert label.shape[0] == 1
            concatenated = torch.cat((img0,img1),0)        
            imshow(torchvision.utils.make_grid(SiameseDataset.inv_normalize(concatenated)).permute(1,2,0).cpu(), 
                                                             f'Label: {label[0].item()}; Dist: {euclidean_distance.item():.2f}',
                                                             save_img=os.path.join(saveimg_folder,f'{i}.png'))

    total_loss = total_loss/len(dataloader)

    accuracy, threshold, roc_auc = compute_stats(np.concatenate(label_, axis=0), np.concatenate(euc_dist_, axis=0))

    return total_loss, accuracy, roc_auc, threshold

def param_optimizer(model_params, data_params, train_params, device):

    def objective(trial):
        fc_dim = trial.suggest_int('fc_dim', 8, 512,log=True)
        out_dim = trial.suggest_int('out_dim', 4, min(fc_dim,32), 4)
        margin = trial.suggest_float('margin', 1, 8)
        lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)

        loss_fn = ContrastiveLoss(margin=margin).to(device)
        net = SiameseNetwork(fc_dim=fc_dim, out_dim=out_dim).to(device)

        train_dataset = SiameseDataset(data_params['train_datapath'], len_factor=data_params['traindata_multfactor'])
        train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=False, num_workers=2, batch_size=train_params['batch_size'])

        val_dataset = SiameseDataset(data_params['val_datapath'], len_factor=data_params['valdata_multfactor'])
        val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=False, num_workers=2, batch_size=train_params['batch_size'])

        optimizer = optim.Adam(itertools.chain.from_iterable([net.parameters(), loss_fn.parameters()]),lr = lr )
        max_val_acc = -math.inf
        for epoch in range(train_params['epochs']):
            epoch_loss = train_epoch(train_dataloader, 
                                     net, 
                                     loss_fn, 
                                     optimizer, 
                                     epoch, 
                                     math.inf, 
                                     None, 
                                     os.path.join(train_params['save_prefix'],train_params['save_model']),
                                     os.path.join(train_params['save_prefix'],train_params['save_loss'])
                                     )
            val_loss, val_acc, val_rocauc, val_thresh = evaluate_loss(val_dataloader, net, loss_fn, device, logwriter=None, x0=epoch*len(val_dataloader))
            print(f"Epoch {epoch}:: train_loss : {epoch_loss:.2f}, val_loss : {val_loss:.2f}, val_acc : {val_acc:.2f}, val_rocauc : {val_rocauc:.2f}, val_thresh: {val_thresh:.1f}")
            if max_val_acc<val_acc:
                max_val_acc = val_acc
        return max_val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def main(model_params, data_params, train_params, device, mode, save_testimg=True):
    loss_fn = ContrastiveLoss().to(device)
    # loss_fn = LearnedLoss().to(device)
    net = SiameseNetwork(**model_params).to(device)
    if train_params['resume_dir']:
        resume_model = natsorted(glob.glob(os.path.join(train_params['resume_dir'],train_params['save_model'].format('*','*'))))[-1]
        resume_loss = re.sub(train_params['save_model'].format('(.*)','(.*)'), train_params['save_loss'].format('\\1','\\2'), resume_model)
        net.load_state_dict(torch.load(resume_model, map_location=device))
        loss_fn.load_state_dict(torch.load(resume_loss, map_location=device))

    if mode == 'train':
        train_dataset = SiameseDataset(data_params['train_datapath'], len_factor=data_params['traindata_multfactor'])
        train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=False, num_workers=2, batch_size=train_params['batch_size'])

        val_dataset = SiameseDataset(data_params['val_datapath'], len_factor=data_params['valdata_multfactor'])
        val_dataloader = DataLoader(val_dataset, shuffle=False, pin_memory=False, num_workers=2, batch_size=train_params['batch_size'])

        optimizer = optim.Adam(itertools.chain.from_iterable([net.parameters(), loss_fn.parameters()]),lr = 0.0005 )
        logwriter = SummaryWriter(train_params['save_prefix'])
        for epoch in range(train_params['epochs']):
            epoch_loss = train_epoch(train_dataloader, 
                                     net, 
                                     loss_fn, 
                                     optimizer, 
                                     device,
                                     epoch, 
                                     train_params['save_freq'], 
                                     logwriter, 
                                     os.path.join(train_params['save_prefix'],train_params['save_model']),
                                     os.path.join(train_params['save_prefix'],train_params['save_loss'])
                                     )
            val_loss, val_acc, val_rocauc, val_thresh = evaluate_loss(val_dataloader, net, loss_fn, device, logwriter=logwriter, x0=epoch*len(val_dataloader))
            print(f"Epoch {epoch}:: train_loss : {epoch_loss:.2f}, val_loss : {val_loss:.2f}, val_acc : {val_acc:.2f}, val_rocauc : {val_rocauc:.2f}, val_thresh: {val_thresh:.1f}")

        logwriter.close()

    test_dataset = SiameseDataset(data_params['test_datapath'], len_factor=data_params['testdata_multfactor'])
    test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=False, num_workers=2, batch_size=1 if save_testimg else train_params['batch_size'])

    test_folder = None
    if save_testimg:
        test_folder = os.path.join(train_params['save_prefix'], 'test_images')
        os.makedirs(test_folder)
    test_loss, test_acc, test_rocauc, test_thresh = evaluate_loss(test_dataloader, net, loss_fn, device, logwriter=None, saveimg_folder = test_folder)

    print(f"test_loss {test_loss:.2f}, test_acc {test_acc:.2f}, test_rocauc {test_rocauc:.2f}, test_thresh {test_thresh:.1f}")



if __name__ == "__main__":

    model_params = {'fc_dim':128, 
                    'out_dim':16}

    # data_params = {'train_datapath':"data/att_faces/Training",
    #                'traindata_multfactor':8,
    #                'val_datapath':"data/att_faces/Testing",
    #                'valdata_multfactor':4,
    #                'test_datapath':"data/att_faces/Testing",
    #                'testdata_multfactor':.2}

    data_params = {'train_datapath':('data/celebA/Img/img_align_celeba','data/celebA/identity_CelebA_filt.train.csv'),
                   'traindata_multfactor':0.05,
                   'val_datapath':('data/celebA/Img/img_align_celeba','data/celebA/identity_CelebA_filt.val.csv'),
                   'valdata_multfactor':1,
                   'test_datapath':('data/celebA/Img/img_align_celeba','data/celebA/identity_CelebA_filt.test.csv'),
                   'testdata_multfactor':1}


    train_params = {'batch_size':648,
                    'resume_dir':None,
                    'save_prefix':"runs/params_optimizer",
                    'epochs':10,
                    'save_freq':1000,

                    'save_model':"siamese_epoch{}_batch{}.pth",
                    'save_loss':"loss_epoch{}_batch{}.pth"
                   }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(prog='main', description='Train/Test Siamese Network')
    parser.add_argument('mode', default='train', choices=['train', 'eval', 'optimize']) 
    args = parser.parse_args()

    os.makedirs(train_params['save_prefix'], exist_ok=True)
    if args.mode == 'optimize':
        param_optimizer(model_params, data_params, train_params, device)
    else:
        torch.save([model_params, data_params, train_params],os.path.join(train_params['save_prefix'],'params.pth'))

        main(model_params, data_params, train_params, device, args.mode)

