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

#average of convolution
for act in secondAct:
    actData['%sAveCov'%act] = names['%s_data'%act].groupby(['userID']).size()/browse_data.groupby(['userID']).size()

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


#cold boot process
def monthStatistics(x):
    valueMonth = list()
    i = 0
    for index ,value in enumerate(x):
        if x[index+1] == 0 and i == 0 :
            continue
        else:
            i = 1
            valueMonth.append(value)
    return valueMonth

for act in mainAct:
    names['%sTimeM'%act] = pd.DataFrame(names['%sTimeM'%act].apply(monthStatistics ,axis=1) ,columns=['%sTimeM'%act])
for act in secondAct:
    names['%sTimeM'%act] = pd.DataFrame(names['%sTimeM'%act].apply(monthStatistics ,axis=1) ,columns=['%sTimeM'%act])
    names['%sTimePre'%act] = pd.DataFrame(names['%sTimePre'%act].apply(monthStatistics ,axis=1) ,columns=['%sTimePre'%act])
#month statitic
for act in mainAct:
    actData['%sTimeLen'%act] = names['%sTimeM'%act].applymap(lambda x: len(x))
    actData['%sTimeSum'%act] = names['%sTimeM'%act].applymap(lambda x: np.sum(x))
    actData['%sTimeMean'%act] = names['%sTimeM'%act].applymap(lambda x: np.mean(x))
    actData['%sTimeStd'%act] = names['%sTimeM'%act].applymap(lambda x: np.std(x)/(np.mean(x)*np.sqrt(len(x))))
    actData['%sTimeSkew'%act] = names['%sTimeM'%act].applymap(lambda x: ss.skew(x))
    actData['%sTimeKurtosis'%act] = names['%sTimeM'%act].applymap(lambda x: ss.kurtosis(x))
    del names['%sTimeM'%act]

for act in secondAct:
    actData['%sTimeLen'%act] = names['%sTimeM'%act].applymap(lambda x: len(x))
    actData['%sTimeSum'%act] = names['%sTimeM'%act].applymap(lambda x: np.sum(x))
    actData['%sTimeStd'%act] = names['%sTimeM'%act].applymap(lambda x: np.std(x)/(np.mean(x)*np.sqrt(len(x))))
    actData['%sTimeSkew'%act] = names['%sTimeM'%act].applymap(lambda x: ss.skew(x))
    actData['%sTimeKurtosis'%act] = names['%sTimePre'%act].applymap(lambda x: ss.kurtosis(x))
    actData['%sTimeLenPre'%act] = names['%sTimePre'%act].applymap(lambda x: len(x))
    actData['%sTimeSumPre'%act] = names['%sTimePre'%act].applymap(lambda x: np.sum(x))
    actData['%sTimeStdPre'%act] = names['%sTimePre'%act].applymap(lambda x: np.std(x)/(np.mean(x)*np.sqrt(len(x))))
    actData['%sTimeSkewPre'%act] = names['%sTimePre'%act].applymap(lambda x: ss.skew(x))
    actData['%sTimeKurtosisPre'%act] = names['%sTimePre'%act].applymap(lambda x: ss.kurtosis(x))
    del names['%sTimeM'%act]

actData['followCount'] = follow_data.groupby(['userID1']).size()
actData['letterCount'] = letter_data.groupby(['userID1']).size()


'''
passive feature
'''
avePost = pd.DataFrame(actData['postTimeMean'].fillna(0))

#user by browsed
browsed_data = pd.merge(browse_data.rename(columns={'blogID':'blogBrowse'}).
                                          drop(['userID' ,'category'] ,axis=1) ,
                       post_data.rename(columns={'blogID':'userPost'}).
                                        drop(['month' ,'category'] ,axis=1) ,
                       left_on='blogBrowse' ,right_on='userPost' ,how='left').drop(['userPost'] ,axis=1)
browsedTimeM = browsed_data.groupby(['userID' ,'month']).size().unstack()
browsedTimeM = browsedTimeM.fillna(0)
browsedTimeM = pd.DataFrame(browsedTimeM.apply(monthStatistics ,axis=1) ,columns=['browsedTimeM'])
actData['browsedTimeTimeLen'] = browsedTimeM.applymap(lambda x: len(x))
actData['browsedTimeTimeSum'] = browsedTimeM.applymap(lambda x: np.sum(x))
actData['browsedTimeTimeMean'] = browsedTimeM.applymap(lambda x: np.mean(x))
actData['browsedTimeTimeStd'] = browsedTimeM.applymap(lambda x: np.std(x)/(np.mean(x)*np.sqrt(len(x))))
actData['browsedTimeTimeSkew'] = browsedTimeM.applymap(lambda x: ss.skew(x))
actData['browsedTimeTimeKurtosis'] = names['browsedTimeM'].applymap(lambda x: ss.kurtosis(x))
#user by lettered 
actData['letteredCount'] = letter_data.groupby(['userID2']).size()
letterAvePost = pd.merge(letter_data ,avePost ,left_on='userID2' ,right_index=True ,how='left').fillna(0)
temp = letterAvePost['postTimeMean'].groupby(letterAvePost['userID2']).mean()
actData['letterAvePost'] =  pd.DataFrame(temp)['postTimeMean']  


del letterAvePost
#user by followed 
actData['followedCount'] = follow_data.groupby(['userID2']).size()
followAvePost = pd.merge(follow_data ,avePost ,left_on='userID2' ,right_index=True ,how='left').fillna(0)
temp = followAvePost['postTimeMean'].groupby(followAvePost['userID2']).mean()
actData['followAvePost'] = pd.DataFrame(temp)['postTimeMean']  
del followAvePost
#user by commented 
commented_data = pd.merge(comment_data.rename(columns={'blogID':'blogComment' ,'userID':'userID1'}).
                                              drop(['category'] ,axis=1) ,
                              post_data.rename(columns={'blogID':'userPost' ,'userID':'userID2'}).
                                        drop(['month' ,'category'] ,axis=1) ,
                       left_on='blogComment' ,right_on='userPost' ,how='left').drop(['userPost'] ,axis=1)
actData['commentedCount'] = commented_data.groupby(['userID2']).size()
commentAvePost = pd.merge(commented_data ,avePost ,left_on='userID1' ,right_index=True ,how='left')
temp = commentAvePost['postTimeMean'].groupby(commentAvePost['userID2']).mean()
actData['commentAvePost'] = pd.DataFrame(temp)['postTimeMean']  
del commentAvePost ,commented_data
#user by voted up 
votedup_data = pd.merge(voteup_data.rename(columns={'blogID':'blogVotedup' ,'userID':'userID1'}).
                                              drop(['category'] ,axis=1) ,
                              post_data.rename(columns={'blogID':'userPost' ,'userID':'userID2'}).
                                        drop(['month' ,'category'] ,axis=1) ,
                       left_on='blogVotedup' ,right_on='userPost' ,how='left').drop(['userPost'] ,axis=1)
votedup_data = votedup_data.drop_duplicates()
actData['votedupCount'] = votedup_data.groupby(['userID2']).size()
votedupAvePost = pd.merge(votedup_data ,avePost ,left_on='userID1',right_index=True ,how='left')
temp = votedupAvePost['postTimeMean'].groupby(votedupAvePost['userID2']).mean()
actData['votedupAvePost'] = pd.DataFrame(temp)['postTimeMean'] 
del votedupAvePost

#user by voted down 
voteddown_data = pd.merge(votedown_data.rename(columns={'blogID':'blogVoteddown' ,'userID':'userID1'}).
                                              drop(['category'] ,axis=1) ,
                              post_data.rename(columns={'blogID':'userPost' ,'userID':'userID2'}).
                                        drop(['month' ,'category'] ,axis=1) ,
                       left_on='blogVoteddown' ,right_on='userPost' ,how='left').drop(['userPost'] ,axis=1)
voteddown_data = voteddown_data.drop_duplicates()
actData['voteddownCount'] = voteddown_data.groupby(['userID2']).size()
voteddownAvePost = pd.merge(voteddown_data ,avePost ,left_on='userID1',right_index=True ,how='left')
temp = voteddownAvePost['postTimeMean'].groupby(voteddownAvePost['userID2']).mean()
actData['voteddownAvePost'] = pd.DataFrame(temp)['postTimeMean'] 

#data store
actData = actData.fillna(0)
actDataScale = actData.apply(lambda x: (x - np.median(x)) / (np.std(x)))
actData.to_csv(data_root+'SMPCUP2017dataset/actStatisticData.csv' ,index_label='userID')
actDataScale.to_csv(data_root+'SMPCUP2017dataset/actStatisticDataScale.csv' ,index_label='userID')