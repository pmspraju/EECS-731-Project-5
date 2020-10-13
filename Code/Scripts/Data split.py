# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:09:21 2020

@author: pmspr
"""
import pandas as pd
from datetime import datetime

#load functions from 
from projectFunctions import loadData

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 7\HW\Git\EECS-731-Project-5\Data'
filename = "Historical Product Demand.csv"
data = loadData(path,filename)

#Remove rows with missing target values
ind = data[data['Date'].isnull()].index.tolist()
data = data.drop(index=ind, axis=0)
data['Date'] = pd.to_datetime(data['Date'])

# ind = data[data['Date'] < pd.Timestamp(2015,12,31)].index.tolist()
# d1 = data.drop(index=ind, axis=0)
# d1.to_csv('2016.csv',index=False)

# ind = data[(data['Date'] < pd.Timestamp(2015,12,31)) | (data['Date'] < pd.Timestamp(2015,12,31))].index.tolist()
# d1 = data[data['Date'] <= pd.Timestamp(2012,12,31)]
# d1.to_csv('2012.csv',index=False)

# d2 = data[(data['Date'] > pd.Timestamp(2012,12,31)) & (data['Date'] <= pd.Timestamp(2013,12,31))]
# d2.to_csv('2013.csv',index=False)

# d3 = data[(data['Date'] > pd.Timestamp(2013,12,31)) & (data['Date'] <= pd.Timestamp(2014,12,31))]
# d3.to_csv('2014.csv',index=False)

# d4 = data[(data['Date'] > pd.Timestamp(2014,12,31)) & (data['Date'] <= pd.Timestamp(2015,12,31))]
# d4.to_csv('2015.csv',index=False)

# d5 = data[(data['Date'] > pd.Timestamp(2015,12,31)) & (data['Date'] <= pd.Timestamp(2016,12,31))]
# d5.to_csv('2016.csv',index=False)

#d6 = data[(data['Date'] > pd.Timestamp(2016,12,31))]
#d6.to_csv('2017.csv',index=False)
