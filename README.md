# EMA_response

The pipline to predict response to EMA prompts is divided into three parts:
1. Feature construction 
2. Loading a pre-trained model 
3. Predicting EMA response and evaluating performance 

Input file: 'EMAresponses_merged.csv'

Table:
1. 'enthusiastic','happy','relaxed','bored','sad','angry','nervous','restless','active','urge': 1 - 5 (likert scale)
2. gender: 'F','M'
3. income: 'low','mid','high','Uknown'
4. days since quit: integer value
5. Age: integer value 
6. prompt.ts: timestamp when EMA was prompted
7. status: 'MISSED', 'ABANDONED_BY_USER', 'ABANDONED_BY_TIMEOUT', 'COMPLETED'
8. user.id: integer value 

Feature construction is performed in the featconst.py script. The script includes methods to perform:
1. Data loading (EMA responses and associated data)
2. Mean imputation of missing EMA responses 
3. Including the variance of responses (over history)
4. Sliding window approach to extract sequences of features of length num_past  

A trained LSTM model can be loaded using the scipt modelLoader.py 

Hyperparameters of the attention LSTM can be set in this script after line 117  














