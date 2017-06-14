#!/usr/bin/env python
#_*_coding:utf-8_*_


"""
从训练集合中抽取数据进行评价
"""


from eval_method import eval 
import numpy as np
import matplotlib.pyplot as plt


def read_question_topic_table(file_qt):
    """
    input: 读取问题话题对照训练集，格式［问题\t话题1，话题2］
    """
    question_labels = []
    for line in open(file_qt):
        line = line.strip()
        segs = line.split('\t')
        try:
            question_id = segs[0]
            labels = segs[1]
            question_labels.append(labels.split(','))
        except Exception as e:
            print(e)
    return question_labels


if __name__ == "__main__":
    # 必须保证顺序是对应的
    real = read_question_topic_table("question_topic_train_set.txt.30w.eval")
    """
    predict = read_question_topic_table("question_topic_train_set.txt.5w")
    for lst in predict:
        for i in range(5 - len(lst)):
            lst.append('000000')
    print(eval(np.array(predict), np.array(real)))
    """
    predict = read_question_topic_table("tmp")
    tmp_test = np.array(predict)[:,0:5]
    res = []
    for i in range(36000):
        i = i + 1
        if i % 1000 == 0:
            print(i)
            res.append(eval(tmp_test[0:i,:], np.array(real[:i])))
    x = np.array(res)
    plt.plot(x)
    plt.show()
