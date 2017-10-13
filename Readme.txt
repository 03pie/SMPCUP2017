Task1:
segmentation.py Make a segmentation of the article
make_idf.py make the idf.txt
tf_idf.py  tfidf analysis

Task2:
gen_mean_lda.py Extract 20 words from each user's blog set via LDA analysis and oo vector averaging of words through word2vec.
keras_classifier.py Train and predict the result by neural network
word2vec_model.py We get the model of word2vec through the trainning of blog sets

Task3:feature_extraction1.pyPositive feature:Main feture: the statistics of post and browseSecond feature: the statistic data of comment amount ,voting up amount, voting down amount ,favoriting amount(All data is divided to twelve parts by month. And the statistics includes all data’s sum ,mean , standard deviation , skew , kurtosis.)Passive feature:The stastics of user has been posted ,browsed ,commented ,voted up ,voted down ,favorited.Feature_extraction2.pyBased on the statistics of dataStatistic1.py’s processing ,dataStatistic2.py added the fist and second half year’s statistic of all data’s sum.xgb_train.pyBased on dataStatistic1.py and dataStatistic2.py , xgb_train.py using processed data to train training set data and forecast validation set data with the xgboost tool