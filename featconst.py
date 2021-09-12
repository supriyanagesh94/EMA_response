#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: supriyanagesh

Feature computation script

Input: 
- .csv file named 'EMAresponses_merged.csv' with the EMA responses, metadata 
- num_past: window size used for making the prediction (integer variable)

This script processes the csv file to extract sequences of length num_past and a corresponding label if the next EMA is completed/not. 
The feature matrix (X), labels (Y), subject ids, EMA time are dumped into a pickle file 

"""
def convert_timestamp(this_time):
    import time 
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(this_time/1000))
    return ts

def computediff(date1,date2,dt_format):
    import datetime 
    diff = datetime.datetime.strptime(date1, dt_format) -datetime.datetime.strptime(date2, dt_format)
    return diff

import numpy as np
import pandas as pd
import datetime 
import pickle
import os 

datapath = '../EMAresponses_merged.csv'
num_past = 5
data = pd.read_csv(datapath)

ema_questions = ['enthusiastic','happy','relaxed','bored','sad','angry','nervous',
                 'restless','active','urge']

num_q_comp = np.zeros((data.shape[0],))
ema_ans = data[ema_questions]

for i in range(data.shape[0]):
    ans = ema_ans.iloc[i,:]
    num_comp = len(ans)-np.sum(np.isnan(ans))
    num_q_comp[i] = num_comp
    
data['num_q_comp'] = num_q_comp

gender = data['gender']
gender = np.array(gender)
gender[gender=='F'] = 1
gender[gender=='M'] = 0

income = data['income']
income = np.array(income)
income[income=='Uknown'] = 0
income[income=='low'] = 1
income[income == 'mid'] = 2
income[income == 'high'] = 3

dsq = data['days since quit']
dsq = np.array(dsq)

age = data['Age']
age = np.array(age)

time_of_day = np.zeros(gender.shape)
day_of_week = np.zeros(gender.shape)
ema_time = data['prompt.ts']
ema_time = np.array(ema_time)
dt_format = "%Y-%m-%d %H:%M:%S"

for i in range(len(ema_time)):
    e_t = convert_timestamp(ema_time[i]*1000)
    e_dt = datetime.datetime.strptime(e_t,dt_format)
    t = e_dt.hour
    d = e_dt.weekday()
    time_of_day[i] = t
    day_of_week[i] = d
    
ema_status_vector = data['status']
ema_status_vector = np.array(ema_status_vector)
ema_status_vector[ema_status_vector == 'MISSED'] = 0
ema_status_vector[ema_status_vector == 'ABANDONED_BY_USER'] = 0
ema_status_vector[ema_status_vector == 'ABANDONED_BY_TIMEOUT'] = 0
ema_status_vector[ema_status_vector == 'COMPLETED'] = 1

subject = data['user.id']
subject = np.array(subject)
subject_list = np.unique(subject)


ema_answers = data[ema_questions]
ema_answers = ema_answers.fillna(0)
ema_answers = np.array(ema_answers)


# IMPUTE MISSING
for i in range(len(subject_list)):
    ind = np.where(subject == subject_list[i])[0]
    sub_mean = np.mean(ema_answers[ind,:],0)
    for j in range(len(ind)):
        this_ind = ind[j]
        if(ema_status_vector[this_ind]==0):
            ema_answers[this_ind,:] = sub_mean
  
# INCLUDING EMA VARIANCE
ema_answers_var = np.empty(ema_answers.shape)
for i in range(len(subject_list)):
    ind = np.where(subject==subject_list[i])[0]
    sub_mean = np.mean(ema_answers[ind,:],0)
    for j in range(len(ind)):
        this_ind = ind[j]
        ema_answers_var[this_ind,:] = (ema_answers[this_ind,:]-sub_mean)**2

ema_answers = np.concatenate((ema_answers,ema_answers_var),axis=1)


prompt_ts = np.array(data['prompt.ts'])

tot = 0
for i in range(len(subject_list)):
    this_subject = subject_list[i]    
    ind = np.where(subject == this_subject)[0]
    if(len(ind)>num_past+1):
        for j in range(0,len(ind)-num_past):
            tot = tot+1

X_sublist = np.empty((tot,1))

X = np.empty((tot,num_past,(11+ema_answers.shape[1])))
delta_T = np.empty((tot,num_past))

Y = np.empty((tot,1))

pid_list = []

count = 0
for i in range(len(subject_list)):
    this_subject = subject_list[i]
    ind = np.where(subject == this_subject)[0]
    this_status = ema_status_vector[ind]
    this_tod = time_of_day[ind]
    this_day = day_of_week[ind]
    this_answers = ema_answers[ind,:]
    this_gender = gender[ind]
    this_age = age[ind]
    this_dsq = dsq[ind]
    this_income = income[ind]
    this_prompt_ts = prompt_ts[ind]
    
    N = len(ind)
    if(N > num_past+1):
        for j in range(N-num_past):
            vec = []
            for k in range(num_past):
                ind_p = j+k

                tod = this_tod[ind_p]
                dw = this_day[ind_p]
                next_tod = this_tod[ind_p+1]
                next_dw = this_day[ind_p+1]
                g = this_gender[ind_p]
                a = this_age[ind_p]
                dq = this_dsq[ind_p]
                inc = this_income[ind_p]
                ema = this_answers[ind_p,:]
                
                if(k==0):
                    pt0 = this_prompt_ts[ind_p]
                    pt0 = convert_timestamp(pt0)
                
                pt = this_prompt_ts[ind_p]
                pt = convert_timestamp(pt)
                
                pt_diff = computediff(pt,pt0,'%Y-%m-%d %H:%M:%S')
                pt_d = pt_diff.days*24 + (pt_diff.seconds/3600)
                
                delta_T[count,k] = pt_d
                
                X[count,k,0] = (tod/24)*5
                X[count,k,1] = (dw/7)*5
                X[count,k,2] = (next_tod/24)*5
                X[count,k,3] = (next_dw/7)*5
                X[count,k,4] = (g/2)*5
                X[count,k,5] = (a/max(age))*5
                X[count,k,6] = (dq/15)*5
                X[count,k,7] = (inc/max(income))*5
                X[count,k,8] = this_status[ind_p]*5
                    
                st_tot = this_status[:ind_p+1]
                cr_tot = len(np.where(st_tot==1)[0])/len(st_tot)            
                X[count,k,9] = cr_tot*5    

                dprev_ind = np.where(this_dsq==dq-1)[0] #dq = today 
                if(len(dprev_ind)>0):
                    st_tot = this_status[dprev_ind]
                    cr_tot = len(np.where(st_tot==1)[0])/len(st_tot)  
                else:
                    cr_tot = 0
                    
                X[count,k,10] = cr_tot*5
                
                X[count,k,11:] = ema  
            
            X_sublist[count,0] = int(this_subject)
            
        Y[count,0] = this_status[ind_p+1]
        pid_list.append(i)
        
        count += 1
        
datasetPath = './data_all/'        
if not os.path.exists(datasetPath):
    os.makedirs(datasetPath)
    
train_name = 'lag_'+str(num_past)+'.pickle'
with open(os.path.join(datasetPath, train_name), 'wb') as f:
    pickle.dump([X, Y, X_sublist,delta_T], f)            
                













