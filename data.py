#!/usr/bin/env python
#_*_coding:utf-8_*_


"""
从训练集合中抽取数据进行评价
"""


from eval_method import eval 
import numpy as np


def read_question_topic_table(file_qt):
    """
    input: 读取问题话题对照训练集，格式［问题\t话题1，话题2］
    """
    question_labels = dict()
    for line in open(file_qt):
        line = line.strip()
        segs = line.split('\t')
        try:
            question_id = segs[0]
            labels = segs[1]
            question_labels[question_id] = labels.split(',')
        except Exception as e:
            print(e)
    return question_labels


if __name__ == "__main__":
    qid_labels = read_question_topic_table(
            "../ieee_zhihu_cup/question_topic_train_set.txt")
    predict = []
    real = []
    for line in open("nohup.out"):
        qid, labels = line.strip().split(':')
        qid = qid.strip()
        labels = labels.strip().split('\t')
        res_lst = [e.split(',') for e in labels]
        forma_lst = [(e[0], int(float(e[1])* (-1.0)))for e in res_lst]
        sorted_fom_lst = sorted(forma_lst, key = lambda x:x[1])
        top5_lst = [e[0] for e in sorted_fom_lst[:5]]
        predict.append(top5_lst)
        real.append(qid_labels[qid])
    print(eval(np.array(predict), np.array(real)))
