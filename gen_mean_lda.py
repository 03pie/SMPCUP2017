import re
import os
from gensim.models import word2vec
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from segmentation import segment, stop_words

if __name__ == '__main__':
    model = word2vec.Word2Vec.load('./model/blogs_word2vec.model').wv
    # LDA analysis for each user's blog set
    user_blogs = sorted(os.listdir('./data/user_blogs/'))
    with open('./data/user_feature_mean.txt', 'w') as fuser_mean:
        for user in user_blogs:
            # for each user
            blogs = pd.read_csv('./data/user_blogs/'+user, header=None, sep='\001', names=['id', 'title', 'text'])
            # Increase the weight of the title and remove the ellipsis
            texts = [re.sub('\.\.+', '.', str(row[1]['title']*6 + row[1]['text']).lower()) for row in blogs.iterrows()]
            corpus = [' '.join(line) for line in segment(texts)]
            # get document-term matrix
            vec = CountVectorizer(stop_words=stop_words)
            vec_tf = vec.fit_transform(corpus)
            # LDA analysis
            n_components = 4
            lda = LatentDirichletAllocation(n_components=n_components, learning_method='batch', max_iter=50, n_jobs=4)
            lda.fit(vec_tf)
            # select 40 words
            words = []
            len_feature = lda.components_.shape[1]
            vec_feature_names = vec.get_feature_names()
            for k in range(n_components):
                word_prob = []
                for i in range(len_feature):
                    word_prob.append((vec_feature_names[i], lda.components_[k][i]))
                word_prob.sort(key=lambda item:item[1], reverse=True)
                words += word_prob[:10]
            words.sort(key=lambda item:item[1], reverse=True)
            words = [term[0] for term in words]
            # words to vector mean
            count = vec_mean = 0
            for word in words:
                if count >= 20:
                    break
                else:
                    try:
                        w_vec = model[word]
                    except Exception:
                        continue
                    else:
                        count += 1
                        vec_mean += w_vec
            vec_mean /= 20
            fuser_mean.write(' '.join([str(v) for v in vec_mean])+'\n')
