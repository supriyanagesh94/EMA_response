import gc
from time import time
import datetime
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


#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

