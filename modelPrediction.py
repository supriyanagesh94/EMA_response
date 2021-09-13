#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: supriyanagesh

- Load the data saved using featconst.py script 
- Loads the trained LSTM model 
- Prediction of next EMA completion for a given input 
- Computes the AUROC, accuracy, confusion matrix for the given X 
"""

from inits import *
from utils import build_model,return_loaded_model,copyModel2Model
from set_hyperparameters import *

num_past = 5
datapath = './data_all/lag_5.pickle'
model_name = 'ema_lstm'
model_path = 'trained_model/'
transfer_upto = "last_dense"

f = open(datapath,'rb')
X, Y, X_sublist,delta_T = pickle.load(f)
f.close()

## Configuration for the LSTM model 
no_feature_cols = X.shape[2]



model = build_model(no_feature_cols=no_feature_cols, output_summary=True,isdropout=isdropout, dropout=dout,rec_dropout=r_dout,time_steps=num_past,attn_type=att,no_cells=cell,lr=lr,optim=optim,att_reg =att_reg,att_kreg=att_kreg,att_areg=att_areg,dropout_layer=dropout_layer,dropout_val=dropout_val,isbidirectional=isbidirectional)

model_s = return_loaded_model(model_name,model_path)
model = copyModel2Model(model_s,model,certain_layer=transfer_upto)

Y_PRED = model.predict(X)
cmat = confusion_matrix(Y,np.around(Y_PRED))
acc = accuracy_score(Y,np.around(Y_PRED))
auc = roc_auc_score(Y,Y_PRED)

print('Accuracy of prediction: ' + str(acc))
print('AUROC score: ' + str(auc))
print('Confusion matrix: ' + str(cmat))
print('CLASSIFICATION REPORT')
print(classification_report(Y, np.around(Y_PRED)))


