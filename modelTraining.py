from inits import *
from utils import build_model,return_loaded_model
from set_hyperparameters import *


model_path = 'trained_model/'
model_name = 'ema_lstm'
num_past = 5

def train(model_name="ema_lstm",attn_type='both',scale_minmax = True,
          balancer=True,batch_size=32, predict=False, return_model=False,write_output=True,no_cells=20,isdropout=True, dropout=0.0,rec_dropout=0.0,
          n_percentage=1.0, time_steps=14, epochs=10, lr=0.001,optim='adam',impute_type='mean', include_cr=False,att_reg=False,att_kreg=0.01,att_areg=0.01,dropout_layer=True,dropout_val=0.5,is_ema_var = True,isbidirectional=True,include_activity=False):


  run_type='train_val'

  datasetPath = 'data_all/'
  log_dir = 'logs/'
  f = open(datasetPath+'lag_'+str(time_steps)+'.pickle','rb')
  X,Y,sub,_ = pickle.load(f)
  f.close()
  sub = np.array(sub)
  sub = np.reshape(sub,(len(sub),1))

  if(run_type=='train_val'):
      sub_list = np.unique(sub[:,0])
      train_perc = 0.7
      val_perc = 0.15
      test_perc = 0.15
      num_train = int(train_perc*len(sub_list))
      num_val = int(val_perc*len(sub_list))
      np.random.seed(5)
      sub_perm = np.random.permutation(len(sub_list))
      sub_train = sub_list[sub_perm[0:num_train]]
      sub_val = sub_list[sub_perm[num_train+1:num_train+num_val]]
      sub_test = sub_list[sub_perm[num_train+num_val+1:]]
      ind_train = []
      for s_train in sub_train:
        ind = np.where(sub[:,0]==s_train)[0]
        ind_train.append(ind)
      ind_val = []
      for s_val in sub_val:
        ind = np.where(sub[:,0]==s_val)[0]
        ind_val.append(ind)
      ind_test = []
      for s_test in sub_test:
        ind = np.where(sub[:,0]==s_test)[0]
        ind_test.append(ind)

      ind_train = np.concatenate(ind_train).ravel()
      ind_val = np.concatenate(ind_val).ravel()
      ind_test = np.concatenate(ind_test).ravel()

      X_TRAIN = X[ind_train,:,:]
      Y_TRAIN = Y[ind_train,:]
      X_VAL = X[ind_val,:,:]
      Y_VAL = Y[ind_val,:]
      X_TEST = X[ind_test,:,:]
      Y_TEST = Y[ind_test,:]

      no_feature_cols = X_TRAIN.shape[2]
      #build model
      model = build_model(no_feature_cols=no_feature_cols, output_summary=True,isdropout=isdropout, dropout=dropout,rec_dropout=rec_dropout, 
                          time_steps=time_steps,attn_type=attn_type,no_cells=no_cells,lr=lr,optim=optim,att_reg =att_reg,att_kreg=att_kreg,att_areg=att_areg,dropout_layer=dropout_layer,dropout_val=dropout_val,isbidirectional=isbidirectional)

      #init callbacks
      tb_callback = TensorBoard(log_dir=log_dir+model_name,
            histogram_freq=0,
            write_grads=False,
            write_images=True,
            write_graph=True)

      checkpoint_dir = model_path + model_name
      #Make checkpoint dir and init checkpointer

      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


      checkpointer = ModelCheckpoint(
        filepath=checkpoint_dir+"/model.{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

      #resampled is false by default
      num_0 = len(np.where(Y_TRAIN==0)[0])
      num_1 = len(np.where(Y_TRAIN ==1)[0])
      wt_0 = (1/num_0)*len(Y_TRAIN)/2
      wt_1 = (1/num_1)*len(Y_TRAIN)/2
      class_weight = {0: wt_0, 1: wt_1}

      #fit
      model.fit(
      x=X_TRAIN,
      y=Y_TRAIN,
      batch_size=batch_size,
      epochs=epochs,
      #callbacks=[tb_callback], #, checkpointer],
      validation_data=(X_VAL, Y_VAL),
      callbacks=[tb_callback],class_weight=class_weight,
      shuffle=True)

      model.save(model_path + model_name + '.h5')

      X_BOOLMAT_VAL = np.zeros(X_VAL.shape,dtype=bool)
      Y_BOOLMAT_VAL = np.zeros(Y_VAL.shape,dtype=bool)
      Y_BOOLMAT_TEST = np.zeros(Y_TEST.shape,dtype=bool)

      if predict:
        print('NUMBER OF LAGS: {0}'.format(time_steps))
        Y_PRED = model.predict(X_VAL)
        Y_PRED = Y_PRED[~Y_BOOLMAT_VAL]
        np.unique(Y_PRED)
        Y_VAL = Y_VAL[~Y_BOOLMAT_VAL]
        Y_PRED_TRAIN = model.predict(X_TRAIN)
        print('Confusion Matrix Validation')
        print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
        print('Validation Accuracy')
        print(accuracy_score(Y_VAL, np.around(Y_PRED)))
        print('ROC AUC SCORE VAL')
        print(roc_auc_score(Y_VAL, Y_PRED))
        print('CLASSIFICATION REPORT VAL')
        print(classification_report(Y_VAL, np.around(Y_PRED)))

        acc_val = accuracy_score(Y_VAL, np.around(Y_PRED))
        auc_val = roc_auc_score(Y_VAL, Y_PRED)
        conf_val = confusion_matrix(Y_VAL, np.around(Y_PRED))

        Y_PRED = model.predict(X_TEST)
        Y_PRED = Y_PRED[~Y_BOOLMAT_TEST]
        np.unique(Y_PRED)
        Y_TEST = Y_TEST[~Y_BOOLMAT_TEST]
        print('Confusion Matrix Test')
        print(confusion_matrix(Y_TEST, np.around(Y_PRED)))
        print('Test Accuracy')
        print(accuracy_score(Y_TEST, np.around(Y_PRED)))
        print('ROC AUC SCORE TEST')
        print(roc_auc_score(Y_TEST,Y_PRED))
        print('CLASSIFICATION REPORT TEST')
        print(classification_report(Y_TEST, np.around(Y_PRED)))

        acc_test = accuracy_score(Y_TEST, np.around(Y_PRED))
        auc_test = roc_auc_score(Y_TEST,Y_PRED)
        conf_test = confusion_matrix(Y_TEST, np.around(Y_PRED))


  if return_model:
    return model



if __name__ == "__main__":

    run_type='train_val' #cv, loso
    scale_minmax = True
    write_output = True
    is_ema_var = True
    impute_type = 'mean'
    include_activity = False


    K.clear_session()
    train(model_name=model_name, epochs=epochs,lr=lr,impute_type=impute_type,isdropout=isdropout,dropout=dout, rec_dropout = r_dout,scale_minmax = scale_minmax,write_output = write_output,batch_size = bs,attn_type = att, predict=True, time_steps=num_past,no_cells=cell,optim=optim,include_cr=True,att_reg =att_reg,att_kreg=att_kreg,att_areg=att_areg,dropout_layer=dropout_layer,dropout_val=dropout_val,is_ema_var = is_ema_var,isbidirectional=isbidirectional,include_activity=include_activity)



