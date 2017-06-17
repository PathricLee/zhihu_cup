import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def zhihu_load_data_and_labels(question_train_set, question_topic_train_set, topic_info):
    """
    目的：加载数据
    question_train_set： 得到一个问题对应的字序列
    question_topic_train_set： 得到一个问题的labels
    topic_info: 话题描述信息
    """
    # topic list
    topic = [line.strip().split('\t')[0] for line in list(open(topic_info, 'r').readlines())]

    # 得到每一个训练实例的标记。对应label置1
    y_labels = []
    for line in open(question_topic_train_set):
        line = line.strip()
        label_str = line.split('\t')[1]
        labels = label_str.split(',')
        index = [0] * 2000
        for label in labels:
            i = topic.index(label)
            index[i] = 1
        y_labels.append(index)
            
    # 得到每一个问题的字符序列。
    x_text = []
    for line in open(question_train_set):
        words = line.split('\t')
        qid, ctitles, wtitles, cdes, wdes = words
        # 字符序列 空格为分割符号
        c_str = ' '.join((','.join((ctitles, cdes))).split(','))
        x_text.append(c_str)

    return [x_text, np.array(y_labels)]
            

def vocab_index(question_train_set, char_embedding, vocab_processor):
    # 得到每一个问题的字符序列。
    """
    max_sen = set()
    for line in open(question_train_set):
        words = line.split('\t')
        qid, ctitles, wtitles, cdes, wdes = words
        # 字符序列 空格为分割符号
        for word in (','.join((ctitles, cdes))).split(',')
            if word not in max_sen:
                max_sen.add(word)
    print(len(max_sen))
    max_sen_str = ' '.join(list(max_sen))
    alist = vocab_processor.transform([max_sen_str])
    """
    pass

    

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    zhihu_load_data_and_labels('question_train_set.txt.5w', 'question_topic_train_set.txt.5w',
            'topic_info.txt')
