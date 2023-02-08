# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 11:14:52 2021

@author: wangm
"""


import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

#from model_rnn import RNNModel
from model_transformer import TransModel
#from utils import load_text_file

import config


def training_k_fold():
    
    split_seed = config.split_seed
    
    data_dict_train, lst_label_train_all = create_list_train()    
    print("split_seed: ", split_seed)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    
    fold = 0
    for train_index, val_index in skf.split(data_dict_train, lst_label_train_all):
        print("Fold : ", fold)
        X_train = [data_dict_train[i] for i in train_index]
        X_val = [data_dict_train[i] for i in val_index]
        Y_train = [lst_label_train_all[i] for i in train_index]
        Y_val = [lst_label_train_all[i] for i in val_index]
        
        print("Y_train len: ", len(Y_train), "Y_val len:", len(Y_val))
        unique, count = np.unique(Y_train, return_counts = True)
        print("Y_train values: ", unique, "count: ", count)
        unique, count = np.unique(Y_val, return_counts = True)
        print("y_val values: ", unique, "count: ", count)
        print("\n")
    
        train_set = BioinformaticsDataset(X_train, Y_train)
        val_set = BioinformaticsDataset(X_val, Y_val)
        
        train_loader = DataLoader(dataset=train_set,batch_size=1,shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=False, num_workers=4)
        
        
        #model = RNNModel()
        model = TransModel()

        
        ckpt_cb = ModelCheckpoint(
            monitor='val_auc', 
            mode='max', 
            dirpath='./space/checkpoints/'+config.model_type+'/'+str(fold), 
            filename='best',
            save_last=False,
            )
        
        es = EarlyStopping(
            monitor='val_auc', 
            patience=config.patience_stop, 
            mode='max',
            )
        
        Logger = TensorBoardLogger(
            save_dir='./space/logs/', 
            name=config.model_type+str(fold),
            )
        
        Callbacks = [es, ckpt_cb]
        
        trainer = pl.Trainer(
            max_epochs=config.epochs_max,
            gpus=config.gpus, 
            accumulate_grad_batches=24,
            #precision=16,
            callbacks=Callbacks,
            logger=Logger,
            distributed_backend=config.distributed_backend,
            num_sanity_val_steps=0,
            # fast_dev_run=True
            )
        
        trainer.fit(model=model,
                    train_dataloader=train_loader,
                    val_dataloaders=val_loader)
        
        fold += 1
    
class BioinformaticsDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        label = self.Y[index]
        inputs = self.X[index]
        
        inputs = torch.from_numpy(inputs.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return inputs, label

    def __len__(self):
        return len(self.X)    
       

def create_list_train():
    list_path_positive_train = glob.glob("TrainDataPSSM/positive/*.pssm") 
    list_path_negative_train = glob.glob("TrainDataPSSM/negative/*.pssm")     
    
    print("Positive train: ", len(list_path_positive_train))
    print("Negative train: ", len(list_path_negative_train))          
    
    list_positive_train_label = [np.ones(1)] *  len(list_path_positive_train)
    list_negative_train_label = [np.zeros(1)] *  len(list_path_negative_train)
    
    list_path_train = list_path_positive_train + list_path_negative_train
    list_label_train = list_positive_train_label + list_negative_train_label
    
    print("Train all: ", len(list_path_train))
    
    data_dict_train = []
    label_dict_train = []
    
    for i, path_train in enumerate(list_path_train):
        if i%1000==0:
            print(i)
            
        with open(path_train) as f:
            lines = f.readlines()
            
        if len(lines)<29:
            continue
            
        start_line = 3
        end_line = len(lines) - 7             
        values = np.zeros((end_line - start_line + 1, 20))
        
        for j in range(start_line, end_line + 1):
            strs = lines[j].strip().split()[2:22]
            for k in range(20):
                values[j-start_line][k] = int(strs[k])
                
        data_dict_train.append(values.astype(np.float32))
        label_dict_train.append(list_label_train[i])
            
    print("")
    return data_dict_train, label_dict_train


if __name__ == "__main__":
    
    print("Adaptor Training K Fold\n")       
    training_k_fold()
