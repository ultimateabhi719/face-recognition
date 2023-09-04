#!/usr/bin/env python3.8
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ResNetForImageClassification


class LearnedLoss(torch.nn.Module):
    """
    Learned loss function
    """

    def __init__(self):
        super(LearnedLoss, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 2)
        )
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, output1, output2, label):
        # label = label.to(torch.int64)
        output = torch.cat((output1,output2),dim=1)
        output = self.fc(output)
        return self.ce_loss(output, label)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = False)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


class SiameseNetwork(nn.Module):
    def __init__(self, model_path = "microsoft/resnet-50", stages_resnet=3, fc_dim=500, out_dim = 32):
        super(SiameseNetwork, self).__init__()
        resnet_cnn = ResNetForImageClassification.from_pretrained(model_path).resnet

        self.pretrained_cnn = nn.Sequential(*([resnet_cnn.embedder]+[resnet_cnn.encoder.stages[s] for s in range(stages_resnet)]) )
        for param in self.pretrained_cnn.parameters():
            param.requires_grad_(False); 

        # self.siamese_cnn = nn.Sequential(*[resnet_cnn.encoder.stages[s] for s in range(stages_resnet,len(resnet_cnn.encoder.stages))])
        self.siamese_cnn = nn.Sequential(*[resnet_cnn.encoder.stages[stages_resnet].layers[0], nn.AdaptiveAvgPool2d(output_size=(1, 1))])

        self.cnn = nn.Sequential(self.pretrained_cnn, self.siamese_cnn)

        self.fc1 = nn.Sequential(
            nn.Linear(2048, fc_dim),
            nn.ReLU(inplace=True),

            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),

            nn.Linear(fc_dim, out_dim))

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


