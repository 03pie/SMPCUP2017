from gensim.models import word2vec
# make a word2vec model
def word2vec_model(blog_seg_path):
    sentences = word2vec.LineSentence(blog_seg_path)
    model = word2vec.Word2Vec(sentences, workers=4)
    return model

if __name__ == '__main__':
    word2vec_model('./data/blog_article_15words_corpus.txt').save("./model/blogs_word2vec.model")
