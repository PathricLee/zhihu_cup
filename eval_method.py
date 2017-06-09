#!/usr/bin/env python
#_*_coding: utf-8 _*_


import numpy as np


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


if __name__ == "__main__":
    predict = np.random.randint(0, 3, size = (10, 5))
    real = np.random.randint(0, 3, size = (10, 4))
    print(predict)
    print(real)
    print(eval(predict, real))
