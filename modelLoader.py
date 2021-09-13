#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: supriyanagesh

Script to load a trained LSTM model given the model path and name 

"""

from inits import *
from utils import return_loaded_model,copyModel2Model,build_model
from set_hyperparameters import *

model_name = 'ema_lstm'
model_path = 'trained_model/'
num_past = 5
datapath = './data_all/lag_5.pickle'
transfer_upto = "last_dense"
## LOAD the data 
f = open(datapath,'rb')
X, Y, X_sublist,delta_T = pickle.load(f)
f.close()

## Configuration for the LSTM model 
no_feature_cols = X.shape[2]


model = build_model(no_feature_cols=no_feature_cols, output_summary=True,isdropout=isdropout, dropout=dout,rec_dropout=r_dout,time_steps=num_past,attn_type=att,no_cells=cell,lr=lr,optim=optim,att_reg =att_reg,att_kreg=att_kreg,att_areg=att_areg,dropout_layer=dropout_layer,dropout_val=dropout_val,isbidirectional=isbidirectional)
model_s = return_loaded_model(model_name,model_path)
model = copyModel2Model(model_s,model,certain_layer=transfer_upto)








