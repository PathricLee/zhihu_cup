#!/usr/bin/env python
#_*_coding: utf-8 _*_


import numpy as np


def eval(predict, la):
    f_lst = []
    weight = 1.0 / np.log([2, 3, 4, 5, 6])
    for ps, qs in zip(predict, la):
        #pred = [1.0 if q == p else 0.0 for q, p in zip(ps, qs)]
        #pred = [1.0 if q in ps else 0.0 for q in qs]
        pred = [1.0 if p in qs else 0.0 for p in ps]

        acc = sum(weight * pred)
        rel = sum(pred) / len(la)
        if acc == 0 or rel == 0:
            f = 0
        else:
            f = acc * rel / (acc + rel)
        f_lst.append(f)
    return sum(f_lst)/len(f_lst)


if __name__ == "__main__":
    predict = np.random.randint(0, 3, size = (10, 5))
    la = np.random.randint(0, 3, size = (10, 4))
    print(predict)
    print(la)
    print(eval(predict, la))
