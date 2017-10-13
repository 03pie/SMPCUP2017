import jieba
import re
# init jieba
jieba.load_userdict("./model/dict.txt")
with open("./model/chinese_stopwords.txt", encoding='utf-8') as fstop:
    stop_words = fstop.read().split()
jieba.enable_parallel(4)
# segmentation
def segment(blogs):
    seg_blogs = [filter([word for word in jieba.cut(article) if word not in stop_words]) for article in blogs]
    return seg_blogs
# filter for words
def filter(tags):
    # remove pure numbers
    tags = [word for word in tags if re.match('^\d+(\.\d+)?$', word) == None]
    # remove substring
    for i in tags:
        for j in tags:
            if i != j and i in j:
                tags.remove(i)
                break
    return tags

if __name__ == "__main__":
    with open("./data/blog_article_original.txt", encoding='utf-8') as blog_in:
        with open("./data/blog_segment.txt", "w", encoding='utf-8') as blog_out:
            blog_out.writelines([' '.join(line) for line in segment(blog_in.readlines())])