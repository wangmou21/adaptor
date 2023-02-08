# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:29:39 2021

@author: wangm
"""

import torch

split_seed = 7

model_type = '3conv20_ln_1Trans128_h5_fc64_weight_maxauc' #'Trans'#'RNN'
# Model parameters
hid_chan = 256   # channel of CNN
kernel_size = 3  # kernel size of CNN
pool_size = 3       # pool size of CNN

hid_gru = 256 # hidden size of GRU
num_gru = 1         # layer number of GRU
dropout = 0         # Dropout in GRU

hid_fc = 512

# train setting
early_stop = True
patience_stop = 20                  # 训练提前终止的epoch数
gpus = [1]            # 所用到的GPU的编号
distributed_backend = "dp"         # 可选ddp, dp 多卡时建议ddp
epochs_max = 100                    # 最大训练的epoch的次数
gradient_clipping = 5.0             # 学习率裁剪的比例 

w_bce = torch.zeros(1)
w_bce[0] = 9.07

# optimizer
optim = 'Adam'
scheduler = True                    # True: ReduceLROnPlateau, None: 
learning_rate = 5e-4                # 初始学习率
patience = 10                   # 学习率裁剪的epoch数
scheduler_factor = 0.5



NUMBER_EPOCHS = 30
LEARNING_RATE = 0.0001

LOSS_WEIGHT_POSITIVE = 10.07
LOSS_WEIGHT_NEGATIVE = 1.11
