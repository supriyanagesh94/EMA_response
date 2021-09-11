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


Feature construction is performed in the featconst.py script. 

