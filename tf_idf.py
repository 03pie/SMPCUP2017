import jieba.analyse
import jieba
import pandas as pd
import re
from segmentation import filter
# tf_idf analysis
def tf_idf(texts):
    jieba.load_userdict("./model/dict.txt")
    jieba.analyse.set_idf_path("./model/idf.txt")
    jieba.analyse.set_stop_words("./model/chinese_stopwords.txt")
    jieba.enable_parallel(8)

    corpus = [filter(jieba.analyse.extract_tags(s, topK = 15)) for s in texts]
    return corpus

if __name__ == '__main__':
    # Enter blog original text
    blogs = pd.read_csv('./data/blog_article_original.txt', header=None, sep='\001', names=['id', 'title', 'text'])
    # Increase the weight of the title and remove the ellipsis
    texts = [re.sub('\.\.+', '.', str(row[1]['title']*6 + row[1]['text']).lower()) for row in blogs.iterrows()]
    # Calculate the topic of each article
    tfidf_corpus = pd.DataFrame(tf_idf(texts))
    # Output the result
    result = pd.DataFrame({'contentid':blogs['id'], 'keyword1':tfidf_corpus[0], 'keyword2':tfidf_corpus[1], 'keyword3':tfidf_corpus[2]})
    result.to_csv('./data/ans_task1.txt', index=None, encoding='utf-8')