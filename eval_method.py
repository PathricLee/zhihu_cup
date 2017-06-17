#!/usr/bin/env python
#_*_coding: utf-8 _*_


import numpy as np
import math


def eval(predict, real):
    f_lst = []
    weight = 1.0 / np.log([2, 3, 4, 5, 6])
    for ps, rs in zip(predict, real):
        #pred = [1.0 if q == p else 0.0 for q, p in zip(ps, qs)]
        #pred = [1.0 if q in ps else 0.0 for q in qs]
        pred = [1.0 if p in rs else 0.0 for p in ps]

        acc = sum(weight * pred)
        rec = sum(pred) / len(rs)
        if acc == 0 or rec == 0:
            f = 0
        else:
            f = acc * rec / (acc + rec)
        f_lst.append(f)
    return sum(f_lst)/len(f_lst)


def eval_office(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]
    :return:
    """
    right_label_num = 0
    right_label_at_pos_num = [0, 0, 0, 0, 0]
    sample_num = 0
    all_marked_label_num = 0
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, len(predict_labels)), predict_labels):
            if label in marked_label_set:
                right_label_num += 1
                right_label_at_pos_num[pos] = right_label_at_pos_num[pos] + 1
        
    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2 + pos)
    recall = float(right_label_num) / all_marked_label_num
    
    
    return (precision * recall) / (precision + recall)

if __name__ == "__main__":
    #predict = np.random.randint(0, 3, size = (10, 5))
    #real = np.random.randint(0, 3, size = (10, 4))
    predict = ([1, 2, 3, 4, 5], (4, 5, 6, 7, 8))
    real = ([4, 5, 6, 7], [4, 5, 6, 7])
    lst = [([1, 2, 3, 4, 5], [4, 5, 6, 7]), ([4, 5, 6, 7, 8], [4, 5, 6, 7])]
    print(eval(predict, real))
    print(eval_office(lst))
