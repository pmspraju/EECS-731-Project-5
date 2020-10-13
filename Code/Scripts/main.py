# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
#import networkx as nx
import numpy  as np
import pandas as pd
import seaborn as sns; sns.set()
import datetime as dt
from matplotlib import pyplot as plt
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, missingValues

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 7\HW\Git\EECS-731-Project-5\Data'

filename = "2016.csv"
data_1 = loadData(path,filename)

filename = "2017.csv"
data_2 = loadData(path,filename)

data_raw = pd.concat([data_1,data_2], ignore_index=True)
data_raw.rename(columns={'Product_Code':'code','Product_Category':'category','Order_Demand':'target'},inplace=True)

#Check the missing values
# misVal, mis_val_table_ren_columns = missingValues(data_raw)
# print(mis_val_table_ren_columns.head(20))

#Remove rows with missing target values
ind = data_raw[data_raw['Date'].isnull()].index.tolist()
data = data_raw.drop(index=ind, axis=0)
#data['target'] = pd.to_numeric(data['target'].str.replace('\(|\)',""),errors='coerce').isnull()
data['target'] = data['target'].str.replace('\(|\)',"")
data['target'] = data['target'].astype('float')
#data = data.astype({"target": float})

#Check the missing values
# misVal, mis_val_table_ren_columns = missingValues(data)
# print(mis_val_table_ren_columns.head(20))

#Compute sum of order values for each ware house
wh_count = pd.DataFrame(data=data,columns=['Warehouse','target'])
group = wh_count.groupby(['Warehouse'], as_index=False)
wh_count = group['target'].sum()
wh_count.rename(columns={'target':'wh_sum'},inplace=True)
#wh_count.to_csv('test.csv',index=False)

#Compute sum of order values for each code 
pc_count = pd.DataFrame(data=data,columns=['code','target'])
group = pc_count.groupby(['code'], as_index=False)
pc_count = group['target'].sum()
pc_count.rename(columns={'target':'pc_sum'},inplace=True)
#pc_count.to_csv('test.csv',index=False) 

#Compute sum of order values for each category
pct_count = pd.DataFrame(data=data,columns=['category','target'])
group = pct_count.groupby(['category'], as_index=False)
pct_count = group['target'].sum()
pct_count.rename(columns={'target':'pct_sum'},inplace=True)
#pct_count.to_csv('test.csv',index=False)  

# teams = data['team1'].unique()

data = pd.merge(data,wh_count, on=['Warehouse'], how='inner')
data = pd.merge(data,pc_count, on=['code'], how='inner')
data = pd.merge(data,pct_count, on=['category'], how='inner')

# from projectFunctions import barPlot, numCount
# barPlot(wh_count['Warehouse'], wh_count['wh_sum'],'Warehouse','Demand','Demand by Warehouse')
# #barPlot(pc_count['code'], pc_count['pc_sum'],'Code','Demand','Demand by Code')
# sns.lineplot(data = pc_count, x="code", y="pc_sum")
# barPlot(pct_count['category'], pct_count['pct_sum'],'Category','Demand','Demand by Product category')

# series = pd.DataFrame(data=data, columns=['Date','target'])
# sns.lineplot(data = series, x="Date", y="target")

from projectFunctions import exploreData, transformData, splitData
features, target = exploreData(data)
features['Date'] = pd.to_datetime(features['Date'])
features['Date'] = features['Date'].dt.dayofyear

#Transform the data
ft, tt = transformData(features, target)

#Check the missing values
# misVal, mis_val_table_ren_columns = missingValues(ft)
# print(mis_val_table_ren_columns.head(20))

X_train, X_test, y_train, y_test = splitData(ft,tt,0.3)

from projectFunctions import lineReg
res_pd = pd.DataFrame([], columns=['Model','AccTrain','AccTest','TrainTime','PredTime'])

results = lineReg(X_train, X_test, y_train, y_test)

# print ("-----------------------------------------------------------------------")
# print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))
# print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
# print ("-----------------------------------------------------------------------")
# res_pd.loc[0,'AccTrain'] = results['acc_train']
# res_pd.loc[0,'AccTest'] = results['acc_test']
# res_pd.loc[0,'TrainTime'] = results['train_time']
# res_pd.loc[0,'PredTime'] = results['pred_time']
# res_pd.loc[0,'Model'] = 'linear'

ft.to_csv('test.csv',index=False)