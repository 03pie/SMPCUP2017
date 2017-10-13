import pandas as pd
from datetime import datetime

import numpy as np
import scipy.stats as ss
from sklearn import preprocessing



data_root = '/media/jyhkylin/本地磁盘1/study/数据挖掘竞赛/SMPCUP2017/'
post_data = pd.read_table(data_root+'SMPCUP2017dataset/2_Post.txt' ,sep='\001' ,names=['userID' ,'blogID' ,'date'])
browse_data = pd.read_table(data_root+'SMPCUP2017dataset/3_Browse.txt' ,sep='\001' ,names=['userID' ,'blogID' ,'date'])
comment_data = pd.read_table(data_root+'SMPCUP2017dataset/4_Comment.txt' ,sep='\001' ,names=['userID' ,'blogID' ,'date'])
voteup_data = pd.read_table(data_root+'SMPCUP2017dataset/5_Vote-up.txt' ,sep='\001' ,names=['userID' ,'blogID' ,'date'])
votedown_data = pd.read_table(data_root+'SMPCUP2017dataset/6_Vote-down.txt' ,sep='\001' ,names=['userID' ,'blogID' ,'date'])
favorite_data = pd.read_table(data_root+'SMPCUP2017dataset/7_Favorite.txt' ,sep='\001' ,names=['userID' ,'blogID' ,'date'])
follow_data = pd.read_table(data_root+'SMPCUP2017dataset/8_Follow.txt' ,sep='\001' ,names=['userID1' ,'userID2'])
letter_data = pd.read_table(data_root+'SMPCUP2017dataset/9_Letter.txt' ,sep='\001' ,names=['userID1' ,'userID2' ,'date'])

names = locals()
mainAct = ['post' ,'browse']
secondAct = ['comment' ,'voteup' ,'votedown' ,'favorite']
relAct = ['follow' ,'letter']
passiveAct = ['browse' ,'comment' ,'voteup' ,'votedown' ,'favorite' ,'follow']
userList = list()
for act in mainAct+secondAct:
    userList= userList + names['%s_data'%act]['userID'].values.tolist() 
userList = list(set(userList))
actData = pd.DataFrame(index=userList)
actData = actData.sort_index()

#user-month form
for act in mainAct+secondAct:
    try:
        names['%s_data'%act]['date'] = names['%s_data'%act]['date'].map(lambda x: datetime.strptime(x ,'%Y-%m-%d %H:%M:%S.0') )
    except:
        try:
            names['%s_data'%act]['date'] = names['%s_data'%act]['date'].map(lambda x: datetime.strptime(x ,'%Y%m%d %H:%M:%S') )
        except:
            names['%s_data'%act]['date'] = names['%s_data'%act]['date'].map(lambda x: datetime.strptime(x ,'%Y-%m-%d %H:%M:%S') )
    names['%s_data'%act]['month'] = names['%s_data'%act]['date'].map(lambda x: x.month)
    names['%s_data'%act] = pd.DataFrame(names['%s_data'%act] ,columns=['userID' ,'blogID' ,'month'])
    names['%s_data'%act]['category'] = act

for act in mainAct+secondAct:
    names['%sTimeM'%act] = names['%s_data'%act].groupby(['userID' ,'month']).size().unstack()
    names['%sTimeM'%act] = names['%sTimeM'%act].fillna(0)

letter_data['date'] = letter_data['date'].map(lambda x: datetime.strptime(x ,'%Y-%m-%d %H:%M:%S.0') )
letter_data['date'] = letter_data['date'].map(lambda x: x.month)
letter_data = letter_data.rename(columns={'date':'month'})
letter_data = letter_data.drop_duplicates()
voteup_data = voteup_data.drop_duplicates()
votedown_data = votedown_data.drop_duplicates()

#month matrix of convolution
for act in secondAct:
    names['%sTimePre'%act] = names['%sTimeM'%act]
for act in secondAct:
    names['%sTimeM'%act] = names['%sTimeM'%act] / browseTimeM
    names['%sTimeM'%act] = names['%sTimeM'%act].dropna(how='all')
    names['%sTimeM'%act] = names['%sTimeM'%act].fillna(0)
    names['%sTimeM'%act][names['%sTimeM'%act]>1] = 1
secondActSumTimeM = commentTimeM + voteupTimeM + votedownTimeM + favoriteTimeM
secondActSumTimeM = secondActSumTimeM.fillna(0)
secondActSumTimeM = commentTimeM.add(secondActSumTimeM ,fill_value=0) + \
 voteupTimeM.add(secondActSumTimeM ,fill_value=0) + votedownTimeM.add(secondActSumTimeM ,fill_value=0) \
 + favoriteTimeM.add(secondActSumTimeM ,fill_value=0)
 
secondActSumTimePre = commentTimePre + voteupTimePre + votedownTimePre + favoriteTimePre
secondActSumTimePre = secondActSumTimePre.fillna(0)
secondActSumTimePre = commentTimePre.add(secondActSumTimePre ,fill_value=0) + \
 voteupTimePre.add(secondActSumTimePre ,fill_value=0) + votedownTimePre.add(secondActSumTimePre ,fill_value=0) \
 + favoriteTimePre.add(secondActSumTimePre ,fill_value=0)
secondAct.append('secondActSum')

#all behavior
mainActSumTimeM = postTimeM + browseTimeM
mainActSumTimeM = mainActSumTimeM.fillna(0)
mainActSumTimeM = postTimeM.add(mainActSumTimeM ,fill_value=0) + browseTimeM.add(mainActSumTimeM ,fill_value=0)
mainAct.append('mainActSum')

allActSumTimeM = secondActSumTimePre + mainActSumTimeM
allActSumTimeM = allActSumTimeM.fillna(0)
allActSumTimeM = mainActSumTimeM.add(allActSumTimeM ,fill_value=0) + secondActSumTimePre.add(allActSumTimeM ,fill_value=0)
mainAct.append('allActSum')


#firt and second half year statistic 
for act in mainAct:
    actData['%sFirstYear'%act] = names['%sTimeM'%act][1]
    actData['%sSecondYear'%act] = names['%sTimeM'%act][7]
    for i in range(2,7):
        actData['%sFirstYear'%act] += names['%sTimeM'%act][i]
        actData['%sSecondYear'%act] = names['%sTimeM'%act][i+6]

for act in secondAct:
    actData['%sFirstYear'%act] = names['%sTimeM'%act][1]
    actData['%sSecondYear'%act] = names['%sTimeM'%act][7]
    for i in range(2,7):
        actData['%sFirstYear'%act] += names['%sTimeM'%act][i]
        actData['%sSecondYear'%act] = names['%sTimeM'%act][i+6]

for act in secondAct:
    actData['%sPreFirstYear'%act] = names['%sTimePre'%act][1]
    actData['%sPreSecondYear'%act] = names['%sTimePre'%act][7]
    for i in range(2,7):
        actData['%sPreFirstYear'%act] += names['%sTimePre'%act][i]
        actData['%sPreSecondYear'%act] = names['%sTimePre'%act][i+6]

actData = actData.fillna(0)
stadata = pd.read_csv(data_root+'SMPCUP2017dataset/actStatisticData.csv').sort_index()
stadata_new = pd.merge(actData ,stadata ,left_index=True ,right_on='userID' ,how='left')
#stadataScale_new = stadata_new.apply(lambda x: (x - np.median(x)) / (np.std(x)))

stadata_new.to_csv(data_root+'SMPCUP2017dataset/actStatisticData_new1.csv' ,index=False)
#stadataScale_new.to_csv(data_root+'SMPCUP2017dataset/actStatisticDataScale_new1.csv' ,index=False)



