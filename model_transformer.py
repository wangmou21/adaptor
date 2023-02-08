# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 19:27:16 2021

@author: wangm
"""

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import math
from sklearn import metrics

import config


LOSS_WEIGHT_POSITIVE = 10.07
LOSS_WEIGHT_NEGATIVE = 1.11
alpha=0.25
gamma=2
def weighted_binary_cross_entropy(output, target, weights=True):        
    if weights :        
        loss = LOSS_WEIGHT_POSITIVE * (target * torch.log(output)) + \
               LOSS_WEIGHT_NEGATIVE * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def calculate_confusion_matrix(arr_labels, arr_labels_hyp):
    corrects = 0
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(arr_labels)):
        confusion_matrix[arr_labels_hyp[i]][arr_labels[i]] += 1

        if arr_labels[i] == arr_labels_hyp[i]:
            corrects = corrects + 1

    acc = corrects * 1.0 / len(arr_labels)
    specificity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    sensitivity = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    mcc = (tp * tn - fp * fn ) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

    return acc, confusion_matrix, sensitivity, specificity, mcc

class TransModel(pl.LightningModule):
      
    def __init__(self):
        super(TransModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3,padding=1)
                
        encoder_layer = nn.TransformerEncoderLayer(d_model=20, nhead=5, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                       
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        
        x = x.transpose(1,2) 
        x = self.conv1(x)
        x = F.layer_norm(x,x.size()[1:])
        x = F.avg_pool1d(x,2)
        x = self.conv2(x)
        x = F.layer_norm(x,x.size()[1:])
        x = F.avg_pool1d(x,2)
        x = self.conv3(x)
        x = F.layer_norm(x,x.size()[1:])        
        x = F.avg_pool1d(x,2)       
        x = x.transpose(1,2) 
        
        x = self.transformer_encoder(x)
        x = F.adaptive_avg_pool2d(x, (1,20))
        
        x = x.squeeze(1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        output = torch.sigmoid(self.fc2(x))
        
        return output
    
    # process inside the training loop
    def training_step(self, batch, batch_idx):
        x , y = batch
        logits = self.forward(x)
        #loss = F.binary_cross_entropy(logits, y, weight=config.w_bce.cuda())  
        loss = weighted_binary_cross_entropy(logits, y)
        #BCE_loss = F.binary_cross_entropy(logits, y, reduce=False)
        #pt = torch.exp(-BCE_loss)
        #loss = alpha * (1-pt)**gamma * BCE_loss

        return loss
    
    # process inside the validation loop
    def validation_step(self, batch, batch_idx):
        x , y = batch
        logits = self.forward(x)
        #val_loss = F.binary_cross_entropy(logits, y, weight=config.w_bce.cuda())  
        val_loss = weighted_binary_cross_entropy(logits, y)
        #BCE_loss = F.binary_cross_entropy(logits, y, reduce=False)
        #pt = torch.exp(-BCE_loss)
        #val_loss = alpha * (1-pt)**gamma * BCE_loss

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
    
        return {'val_loss': val_loss, 
                'logits': logits.squeeze(1),
                'y': y.squeeze(1).int(),
                }    
    

    def validation_epoch_end(self, outputs):
        
        logits_all = torch.stack([x['logits'] for x in outputs])
        y_all = torch.stack([x['y'] for x in outputs])
        
        pred_all = logits_all.round().int()
        
        logits_all = logits_all.cpu().numpy()
        y_all = y_all.cpu().numpy()
        pred_all = pred_all.cpu().numpy() 

        val_auc = metrics.roc_auc_score(y_all, logits_all)
        val_avg_acc, confusion_matrix, val_sensitivity, val_specificity, val_mcc = calculate_confusion_matrix(y_all, pred_all)       

        self.log('val_avg_acc',val_avg_acc)
        self.log('val_auc',val_auc)
        self.log('val_sensitivity',val_sensitivity)
        self.log('val_specificity',val_specificity)
        self.log('val_mcc',val_mcc)

    # process inside the test loop
    def test_step(self, batch, batch_idx):
        x , y = batch
        logits = self.forward(x)
   
        return {'logits': logits.squeeze(1),
                'y': y.squeeze(1).int(),
                }    
    
    def test_epoch_end(self, outputs):
        
        logits_all = torch.stack([x['logits'] for x in outputs])
        y_all = torch.stack([x['y'] for x in outputs])
        
        pred_all = logits_all.round().int()
        
        logits_all = logits_all.cpu().numpy()
        y_all = y_all.cpu().numpy()
        pred_all = pred_all.cpu().numpy() 
        


        test_auc = metrics.roc_auc_score(y_all, logits_all)
        
        test_avg_acc, confusion_matrix, test_sensitivity, test_specificity, test_mcc = calculate_confusion_matrix(y_all.squeeze(), pred_all.squeeze())       

        print('val_avg_acc:',test_avg_acc)
        print('val_auc:',test_auc)
        print('val_sensitivity:',test_sensitivity)
        print('val_specificity:',test_specificity)
        print('val_mcc:',test_mcc)
        
        return {'logits_all': logits_all,
                'y_all': y_all,
                }
    
    def configure_optimizers(self):
        
        if config.optim == 'Adam':
           optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        elif config.optim == 'AdamW':    
           optimizer = optim.AdamW(lr=1e-3, params=self.parameters())
               
        if config.scheduler is None:
            return optimizer
        else:       
            return {
                    'optimizer': optimizer,
                    'lr_scheduler': ReduceLROnPlateau(
                        optimizer, mode='min', factor=config.scheduler_factor, patience=config.patience, verbose=True, min_lr=1e-5),
                    'monitor': "val_loss"
                    }
        
