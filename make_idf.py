from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from segmentation import segment, stop_words
# make_Tfidfvectorizer
def make_idf(corpus):
	vectorizer = TfidfVectorizer(stop_words=stop_words)
	vectorizer.fit_transform(corpus)
	return vectorizer

if __name__ == '__main__':
    with open("./data/blog_article_original.txt", "r", encoding='utf-8') as fblog:
        text = fblog.readlines()
    # segmentation
    corpus = [' '.join(line) for line in segment(text)]
    # make idf.txt
    vec = make_idf(corpus)
    pd.DataFrame({'col1':vec.get_feature_names(), 'col2':vec.idf_}).to_csv("./model/idf.txt", encoding='utf-8', sep=' ', index=None, header=None)
