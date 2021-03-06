#!/usr/bin/env python
#_*_coding:utf-8_*_


"""
从训练集合中抽取数据进行评价
"""


from eval_method import eval 
from eval_method import eval_office
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
    real = read_question_topic_table("question_topic_train_set.txt.5w")
    """
    predict = read_question_topic_table("question_topic_train_set.txt.5w")
    for lst in predict:
        for i in range(5 - len(lst)):
            lst.append('000000')
    print(eval(np.array(predict), np.array(real)))
    """
    predict = read_question_topic_table("predict.txt.train.150w")
    res = []
    for i in range(len(predict)):
        i = i + 1
        if i % 1000 == 0:
            res.append(eval(np.array(predict[:i]), np.array(real[:i])))
    res.append(eval(np.array(predict), np.array(real)))
    x = np.array(res)
    plt.plot(x)
    plt.show()
    """
    res = []
    for_eval = []
    for x, y in zip(predict, real):
        e = (x, y)
        for_eval.append(e)
    for i in range(len(predict)):
        i += 1
        if i % 1000 == 0:
            res.append(eval_office(for_eval[:i]))
    res.append(eval_office(for_eval))
    x = np.array(res)
    plt.plot(x)
    plt.show()
    """
