# _*_ coding: utf-8 _*_
# @Time    :   2023/12/29 15:22:14
# @FileName:   word2vector.py
# @Author  :   wyx
# @Email   :   weiyyuxin@163.com & weiyyuxin@gmail.com
# @Software:   VSCode

import torch.nn as nn 
import pandas as pd
import jieba 
import re
import collections
from sklearn.decomposition import PCA    


class Word2Vector(nn.Module):
    def __init__(self):
        super(Word2Vector).__init__()
        self.doc = [] # 对应数据集所有单词
        self.window_size = 5
        self.min_count = 5
        self.word2count = {}
        self.word2word_matrix = []
        self.word2word_dict = {}
        self.word_dim = 200
        self.text_to_words()
        self.build_windows()
        self.build_word2count()
        self.build_word_to_word_matrix()
        
    def text_to_words(self):   # 这个方法根据自己数据情况修改，数据集太大的话，会很慢，根据数据情况调整
        for line in open(r"../toutiao_cat_data.txt"):
            tmp = line.split("_!_")
            temp =jieba.lcut(tmp[3]) 
            words = []
            for i in temp:
                i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
                if len(i) > 0:
                    words.append(i)
            if len(words) > 0:
                self.doc.append(words)
            if len(tmp) > 4: # 编号_!_序号_!_类别_!_文本_!_文本
                i = 4
                while i < len(tmp):
                    temp =jieba.lcut(tmp[i]) 
                    words = []
                    for j in temp:
                        j = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", j)
                        if len(j) > 0:
                            words.append(j)
                    if len(words) > 0:
                        self.doc.append(words)
                    i += 1
    # 构建字典,Python中的字典应用了哈希表的原理
    def build_word2count(self):
        words = []
        for data in self.doc:
            words.extend(data)
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= self.min_count]
        self.word2count= {item[0]:item[1] for item in reserved_words}
    
    # 构造上下文窗口，CBOW用上下文词预测中心词
    def build_windows(self):
        for index, data in enumerate(self.doc):
            context = []
            for ids in range(len(data)):
                if ids < self.window_size:
                    left = data[:ids]
                else:
                    left = data[ids - self.window_size:ids]
                if ids + self.window_size > len(data):
                    right = data[ids + 1:]
                else:
                    right = data[ids + 1:ids + self.window_size + 1]
                context = left + [data[ids]] + right
                # print(context)
                for word in context:
                    if word not in self.word2word_dict:
                        self.word2word_dict[word] = {}
                    for co_word in context:  # 双层遍历
                            if co_word != word:   # 比如 co_word: 京城；word：最
                                if co_word not in self.word2word_dict[word]:
                                    self.word2word_dict[word][co_word] = 1
                                else:
                                    self.word2word_dict[word][co_word] += 1  
        return self.word2word_dict
    
    # 构建词和词的共现矩阵
    def build_word_to_word_matrix(self):
        word_list = list(self.word2count)
        words_all = len(word_list)
        count = 0
        for w in word_list:
            count += 1
            print(count, "/", words_all)
            tmp = []
            sum_count = sum(self.word2word_dict[w].values()) # w 和 某个词一起出现的次数
            for co_w in word_list:
                if(sum_count == 0):
                    print(w)
                    continue
                weight = self.word2word_dict[w].get(co_w, 0) / sum_count
                # print(w)
                # print(co_w)
                # print(word2word_dict[w].get(co_w, 0))
                # print(weight)
                tmp.append(weight)
            self.word2word_matrix.append(tmp)
            
        return self.word2word_matrix
    
    # 使用PCA降维
    def low_dimension(self):
        pca = PCA(n_components=self.word_dim)  # 这里的word_dim维度没有变化，可以自己设置值
        low_emb = pca.fit_transform(self.word2word_matrix)
        return low_emb
    
    # 保存模型，存的是embedding 
    def save_model(self): 
        word_embedding_dict = {index: embedding for index, embedding in enumerate(self.low_dimension())}
        ids_to_word = {index: word for index, word in enumerate(list(self.word2count))}
        with open("word2vector_model.bin",'w+') as f:
            for w_ids, w_emb in word_embedding_dict.items():
                # print(w_emb)
                word = ids_to_word[w_ids]
                word_embedding = [str(item) for item in w_emb]
                f.write(word + '\t' + ','.join(word_embedding) + '\n')
        f.close()

vec = Word2Vector()
vec.save_model()