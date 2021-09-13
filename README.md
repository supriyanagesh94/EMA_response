# EMA_response

The pipline to predict response to EMA prompts is divided into three parts:
1. Feature construction 
2. Loading a pre-trained model 
3. Predicting EMA response and evaluating performance 

Input file: 'EMAresponses_merged.csv'

The .csv file should contain EMA responses and other related information corresponding to each prompt. The column names and corresponding values are listed here. 

| Column name  | Acceptable values  |  
|---|---|
| 'enthusiastic','happy','relaxed','bored','sad','angry','nervous','restless','active','urge'  | 1 - 5 (Likert scale) |  
| 'gender'  | 'F','M' |  
| 'income'  | 'low','mid','high','Uknown'  |  
| 'days since quit' | integer value |
| 'Age' | integer value |
| 'prompt.ts'  | Unix timestamp when EMA was prompted  |
| 'status'  | 'MISSED', 'ABANDONED_BY_USER', 'ABANDONED_BY_TIMEOUT', 'COMPLETED'  |
| 'user.id' | integer value |


Feature construction is performed in the featconst.py script. The script includes methods to perform:
1. Data loading (EMA responses and associated data)
2. Mean imputation of missing EMA responses 
3. Including the variance of responses (over history)
4. Sliding window approach to extract sequences of features of length num_past  
5. Saving the the constructed train, val, test sequences in .pickle files 

Training an LSTM model from scratch - using the script modelTraining.py
This script will load the data from the saved .pickle files, initialize and train an attention LSTM model

A trained LSTM model can be loaded using the scipt modelLoader.py 

Hyperparameters of the attention LSTM can be set in this script after line 117  

The model performance for prediction can be computed using the script modelPrediction.py. The script includes methods to perform:

1. Loading the saved features from featconst.py
2. Loading the trained model 
3. Generating the predictions for next EMA response for each input sequence 
4. Computing the AUROC, Accuracy, Confusion matrix for the predictions


# To generate results with a new EMA dataset (using a pre-trained LSTM model) 

1. Get the data in a .csv file with the row corresponding to one EMA prompt. The csv file should have information as shown in the table and the exact same column names.  
2. Save the csv file as 'EMAresponses_merged.csv' in the home directory 
3. Run _featconst.py_ - this will create sequences of length 5 and save the sequence of features and labels in the file _lag_5.pickle_
4. Run _modelPrediction.py_ to load a trained lstm model from the directory _trained_model/_ and load the saved data in _lag_5.pickle_. The accuracy, AUROC score and confusion matrix for the prediction is printed. 

# To train an LSTM model from scratch 
1. Get the data in a .csv file with the row corresponding to one EMA prompt. The csv file should have information as shown in the table and the exact same column names.
2. Save the csv file as 'EMAresponses_merged.csv' in the home directory
3. Run _featconst.py_ - this will create sequences of length 5 and save the sequence of features and labels in the file _lag_5.pickle_
4. Run modelTraining.py and set a new model_name in line 270 (example: ema_lstm2)
5. Run _modelPrediction.py_ with the changed model_name in line 139. The model performance is printed on screen.   







