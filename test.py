# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:33:05 2021

@author: wangm
"""



import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn import metrics
import matplotlib.pyplot as plt

from torchsummaryX import summary

from train import BioinformaticsDataset
from model_transformer import TransModel, calculate_confusion_matrix

import config

del_min = False


def test_all():
    
    data_dict_test, lst_label_test = create_list_test()  
    
    test_set = BioinformaticsDataset(data_dict_test, lst_label_test)
    test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False, num_workers=1)
    
    logits_all_fold = []

    for fold in range(5):
        
        logits_all = []         
        y_all = []  
        mmc_all = []
        checkpoint_path = './space/checkpoints/'+config.model_type+'/'+str(fold)+'/best.ckpt'
    
        model = TransModel()
        summary(model,x=torch.zeros(1,100,20))
        model = model.load_from_checkpoint(checkpoint_path)
        model.cuda()
        model.eval()
        
        for i,data in enumerate(test_loader):
            inputs,label = data
            inputs = inputs.cuda()
            with torch.no_grad():
                logits = model(inputs)
            logits_all.append(logits.squeeze(1))
            y_all.append(label.squeeze(1).int())    

        logits_all = torch.stack(logits_all)
        y_all = torch.stack(y_all)
        
        logits_all = logits_all.cpu().numpy().squeeze()
        y_all = y_all.cpu().numpy().squeeze()
        
        test_auc = metrics.roc_auc_score(y_all, logits_all)
        
        test_avg_acc, confusion_matrix, test_sensitivity, test_specificity, test_mcc = calculate_confusion_matrix(y_all, logits_all.round().astype(np.int32))       
        
        print('Fold:', fold)
        print('val_avg_acc:',test_avg_acc)
        print('val_auc:',test_auc)
        print('val_sensitivity:',test_sensitivity)
        print('val_specificity:',test_specificity)
        print('val_mcc:',test_mcc)
        
        logits_all_fold.append(logits_all)
        mmc_all.append(test_mcc)
     
    print('Ensemble')
    if del_min:
        index_min = np.argmin(mmc_all)
        logits_all_fold.pop(index_min)
    logits_all_fold = np.array(logits_all_fold).mean(0)
    test_auc = metrics.roc_auc_score(y_all, logits_all_fold)
    test_precision, test_recall, _thresholds = metrics.precision_recall_curve(y_all, logits_all_fold)
    test_auprc = metrics.auc(test_recall, test_precision)
    test_avg_acc, confusion_matrix, test_sensitivity, test_specificity, test_mcc = calculate_confusion_matrix(y_all, logits_all_fold.round().astype(np.int32))       
    print('val_avg_acc:',test_avg_acc)
    print('val_auc:',test_auc)
    print('val_auprc:',test_auprc)
    print('val_sensitivity:',test_sensitivity)
    print('val_specificity:',test_specificity)
    print('val_mcc:',test_mcc)
    
    #fpr, tpr, thersholds = metrics.roc_curve(y_all, logits_all_fold)
    
    #plt.rc('font',family='Times New Roman') 
    #plt.plot(fpr, tpr, 'b', label='ROC curve (area = {0:.4f})'.format(test_auc), lw=2)
 
    #plt.xlim([-0.05, 1.05])  
    #plt.ylim([-0.05, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')  
    #plt.title('Receiver Operating Characteristic')
    #plt.legend(loc="lower right")
    #plt.savefig("roc.png", dpi=600)
    #plt.show()

    
    plt.rc('font',family='Times New Roman') 
    plt.plot(test_precision, test_recall, 'b', label='ROC curve (area = {0:.4f})'.format(test_auprc), lw=2)
 
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')  
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig("prc.png", dpi=600)
    plt.show()

def create_list_test():
    list_path_positive_test = glob.glob("TestDataPSSM/positive/*.pssm") 
    list_path_negative_test = glob.glob("TestDataPSSM/negative/*.pssm")     
    
    print("Positive test: ", len(list_path_positive_test))
    print("Negative test: ", len(list_path_negative_test))          
    
    list_positive_test_label = [np.ones(1)] *  len(list_path_positive_test)
    list_negative_test_label = [np.zeros(1)] *  len(list_path_negative_test)
    
    list_path_test = list_path_positive_test + list_path_negative_test
    list_label_test = list_positive_test_label + list_negative_test_label
    
    print("Test all: ", len(list_path_test))
    
    data_dict_test = []
    label_dict_test = []
    
    for i, path_test in enumerate(list_path_test):
            
        with open(path_test) as f:
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
                
        data_dict_test.append(values.astype(np.float32))
        label_dict_test.append(list_label_test[i])
            
    print("")
    return data_dict_test, label_dict_test

if __name__ == "__main__":
          
    test_all()
