#!/usr/bin/env python
#_*_coding:utf-8_*_


import numpy as np
import time


def read_question_topic_table(file_qt):
    """
    input: 读取问题话题对照训练集，格式［问题\t话题1，话题2］
    output: label对应的问题集合，以及其先验概率,次数
    """
    label_question = dict()
    label_prob = dict()
    for line in open(file_qt):
        line = line.strip()
        segs = line.split('\t')
        try:
            question_id = segs[0]
            labels = segs[1]
            for label in labels.split(','):
                label_question.setdefault(label, []).append(question_id)
        except Exception as e:
            print(e)
    count_ci = [len(lst) for label, lst in label_question.items()]
    pro_ci = np.array(count_ci) / float(sum(count_ci))
    pro_ci = list(pro_ci)
    for i, (label, lst) in enumerate(label_question.items()):
        label_prob[label] = [pro_ci[i], count_ci[i]]
    """
    for label, prob in label_prob.items():
        print("%s\t%s" % (label, prob))
    """

    return label_question, label_prob


def question_info(file_question):
    """
    从每一个问题的描述中读取, 现在只看词
    """
    question_words = dict()
    for i, line in enumerate(open(file_question)):
        line = line.strip()
        words = line.split('\t')
        wtitles = ""
        wdes = ""
        if len(words) == 5:
            qid, ctitles, wtitles, cdes, wdes = words
        elif len(words) == 3:
            qid, ctitles, wtitles = words
        else:
            qid = words[0]
        all_words_str = "%s,%s" % (wtitles, wdes)
        words_all = all_words_str.split(',')
        question_words[qid] = words_all
    return question_words
    

def main():
    label_question, label_pro = read_question_topic_table("../ieee_zhihu_cup/question_topic_train_set.txt")
    #question_words = question_info("../ieee_zhihu_cup/question_train_set.txt") 
    question_words = question_info("../cnn-text-classification-tf/question_train_set.txt.10w") 
    label_word_times = dict()
    word_set = set()
    start = time.clock()
    try:
        for label, question_lst in label_question.items():
            for question in question_lst:
                if question not in question_words:
                    continue
                for word in question_words[question]:
                    key = "%s\t%s" % (label, word)
                    label_word_times[key] = label_word_times.get(key, 1) + 1
                    word_set.add(word)
    except Exception as e:
        print(e)
    print(len(word_set))
    return
    # 测试
    for i, (qid, words) in enumerate(question_words.items()):
        res = []
        for label in label_question:
            log_evl = np.log(label_pro[label][0])
            for word in word_set:
                key = "%s\t%s" % (label, word)
                num = label_word_times.get(key, 1)
                prob = float(num) / label_pro[label][1]
                log_evl += np.log(prob)
            cost = int(log_evl * (-1))
            res.append((label, cost))

        # sort
        res_sort = sorted(res, key = lambda x:x[1])
        x = [label for label, cost in res_sort[:5]]
        if i == 1000:
            break
    return res_sort, 


def test():
    label_question, label_pro = read_question_topic_table("../ieee_zhihu_cup/question_topic_train_set.txt")
    for label, qid_lst in label_question.items():
        print("%s\t%s" % (label, len(qid_lst)))


if __name__ == "__main__":
    main()
