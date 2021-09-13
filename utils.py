from inits import *


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

  if(isbidirectional):
      x = Bidirectional(LSTM(no_cells,dropout=dropout,recurrent_dropout=rec_dropout))(x)
  else:
      x = LSTM(no_cells,dropout=dropout, recurrent_dropout=rec_dropout)(x)

  preds = Dense(10,activation='relu')(x)
  if(dropout_layer):
      preds = Dropout(dropout_val)(preds)
  preds = Dense(1,activation='sigmoid')(preds)
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


def convert_timestamp(this_time):
    import time
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(this_time/1000))
    return ts

def computediff(date1,date2,dt_format):
    import datetime
    diff = datetime.datetime.strptime(date1, dt_format) -datetime.datetime.strptime(date2, dt_format)
    return diff
