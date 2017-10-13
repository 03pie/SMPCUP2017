from collections import Counter

import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold 
import matplotlib.pyplot as plt

import numpy as np

data_root = '/media/jyhkylin/本地磁盘1/study/数据挖掘竞赛/SMPCUP2017/'
train3 = pd.read_table(data_root+'SMPCUP2017dataset/SMPCUP2017_TrainingData_Task3.txt' 
                       ,sep='\001' ,names=['userID' ,'growthValue'])
stadata3 = pd.read_csv(data_root+'SMPCUP2017dataset/actStatisticData_new1.csv')
train3 = pd.merge(train3 ,stadata3 ,left_on='userID' ,right_on='userID' ,how='left')
x = np.array(train3.drop(['userID' ,'growthValue'] ,axis=1))
y = np.array(train3['growthValue'])

#submit
test3 = pd.read_table(data_root+'SMPCUP2017dataset/SMPCUP2017_TestSet_Task3.txt' ,
                    sep='\001' ,names=['userID'])
test3 = pd.merge(test3 ,stadata3 ,left_on='userID' ,right_on='userID' ,how='left')
param = {'max_depth':10, 
             'eta': 0.22, 
             'silent': 1, 
             'objective': 'reg:tweedie', 
             'booster': 'gbtree' ,
             'seed':10 ,
             'base_score':0.5 ,
             'eval_metric':'mae' ,
             'min_child_weight':1 ,
             'gamma':0.007 ,
             'tree_method':'hist' ,
             'tweedie_variance_power':1.54 ,
             'nthread':4
             }
num_round = 45
dtrain = xgb.DMatrix(x,label=y)
bst = xgb.train(param, dtrain, num_round)

x_t = xgb.DMatrix(np.array(test3.drop(['userID'] ,axis=1)))
y_t = bst.predict(x_t)
task3 = pd.DataFrame([test3['userID'] ,y_t]).T
task3 = task3.rename(columns={'userID':'userid' ,'Unnamed 0':'growthvalue'})
task3.to_csv('task3_final.txt' ,index=False ,sep=',')















































