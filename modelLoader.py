#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: supriyanagesh

Script to load a trained LSTM model given the model path and name 

"""
import gc
from time import time
import os
import math
import pickle

import numpy as np
import pandas as pd
#from pad_sequences import PadSequences
#from processing_utilities import PandasUtilities
from attention_function import attention_3d_block as Attention_time
from attention_function import attention_3d_block_time_features as Attention #attention_3d_block_time_features(inputs, TIME_STEPS):
from attention_function import attention_spatial_block as Attention_feat #attention_spatial_block(inputs):
#from attention_function import attention_3d_block_reg as Attention_reg#attention_3d_block_reg(inputs, TIME_STEPS,kreg,areg)
from attention_function import attention_time_reg as Attention_reg

from keras import backend as K
from keras.models import Model, Input, load_model #model_from_json
from keras.layers import Masking, Flatten, Embedding, Dense,Dropout, LSTM, TimeDistributed,Softmax,Bidirectional
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import optimizers


def return_loaded_model(model_name="ema_lstm", model_path='saved_models/'):

    loaded_model = load_model(model_path+model_name+'.h5')
    return loaded_model

def copyModel2Model(model_source,model_target,certain_layer=""):
    for l_tg,l_sr in zip(model_target.layers,model_source.layers):
        wk0=l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name==certain_layer:
            break
    print("model source was copied into model target")
    return model_target

def build_model(no_feature_cols=None, time_steps=7, output_summary=False,no_cells=256,lr=0.001,attn_type='both',
        isdropout=True, dropout=0.0, rec_dropout=0.0,optim='adam',att_reg=False,att_kreg=0.01,att_areg=0.01,dropout_layer=True,dropout_val=0.5,isbidirectional=True):

  """

  Assembles RNN with input from return_data function

  Args:
  ----
  no_feature_cols : The number of features being used AKA matrix rank
  time_steps : The number of days in a time block
  output_summary : Defaults to False on returning model summary

  Returns:
  ------- 
  Keras model object

  """
  if(not isdropout):
      dropout = 0.0
      rec_dropout= 0.0

  print("time_steps:{0}|no_feature_cols:{1}".format(time_steps,no_feature_cols))
  input_layer = Input(shape=(time_steps, no_feature_cols))
  if(att_reg):
    x = Attention_reg(input_layer,time_steps,att_kreg,att_areg)
  else:
    if(attn_type=='both'):
        x = Attention(input_layer, time_steps)
    elif(attn_type == 'time'):
        x = Attention_time(input_layer, time_steps)
    elif(attn_type == 'feat'):
        x = Attention_feat(input_layer, time_steps)

  #x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(x)
  if(isbidirectional):
      x = Bidirectional(LSTM(no_cells,dropout=dropout,recurrent_dropout=rec_dropout,name="lstm"))(x)
  else:
      x = LSTM(no_cells,dropout=dropout, recurrent_dropout=rec_dropout,name="lstm")(x)

  preds = Dense(10,activation='relu')(x)
 # preds = Dense(5)(preds)
  if(dropout_layer):
      preds = Dropout(dropout_val)(preds)
  preds = Dense(1,activation='sigmoid',name="last_dense")(preds)
  #preds = Softmax()(preds)
  model = Model(inputs=input_layer, outputs=preds)

  #RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
  if(optim=='rms'):
    RMS = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08)
  elif(optim=='adam'):
    RMS = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99, amsgrad=False)
  elif(optim=='sgd'):
    RMS = optimizers.SGD(lr=lr,momentum=0.0,decay=0.0,nesterov=False)

  model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])

  if output_summary:
    model.summary()
  return model


datapath = './data_all/lag_5.pickle'
## LOAD the data 
f = open(datapath,'rb')
X, Y, X_sublist,delta_T = pickle.load(f)
f.close()

## Configuration for the LSTM model 
no_feature_cols = X.shape[2]
isdropout = False
dropout = 0
time_steps = 5 
no_cells = 30
optim = 'adam'
attn_type = 'feat'
lr = 1e-5 
att_reg = False 
att_kreg = 0.01 
att_areg = 0.05 
dropout_layer = False 
dropout_val = 0.5 
rec_dropout = 0
isbidirectional = True 
transfer_upto = "last_dense"

TL_modelname = 'ema_lstm'
TL_modelpath = 'train_model/'

model = build_model(no_feature_cols=no_feature_cols, output_summary=True,isdropout=isdropout, dropout=dropout,rec_dropout=rec_dropout,
                          time_steps=time_steps,attn_type=attn_type,no_cells=no_cells,lr=lr,optim=optim,att_reg =att_reg,att_kreg=att_kreg,att_areg=att_areg,dropout_layer=dropout_layer,dropout_val=dropout_val,isbidirectional=isbidirectional)
model_s = return_loaded_model(TL_modelname,TL_modelpath)
model = copyModel2Model(model_s,model,certain_layer=transfer_upto)








