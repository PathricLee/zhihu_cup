import numpy
import sys
import gensim
import cPickle

chars_title_emb_avg={}
words_title_emb_avg={}
chars_descr_emb_avg={}
words_descr_emb_avg={}

chars_em=gensim.models.KeyedVectors.load_word2vec_format("char_embedding.txt",binary=False)
words_em=gensim.models.KeyedVectors.load_word2vec_format("word_embedding.txt",binary=False)
def get_avg_emb(char_list,em_type):
    chars_ems=[]
    for i in char_list:
        try:
            i_em=em_type[i]
            chars_ems.append(i_em)
        except:
            sys.stderr.write(i+"\n")
    chars_ems_avg=numpy.mean(numpy.array(chars_ems),axis=0)

for line in sys.stdin:
    line=line.strip()
    ss=line.split("\t")
    if len(ss)<4:
        continue

    chars_title=ss[1].split(",")
    words_title=ss[2].split(",")
    chars_descr=ss[3].split(",")
    words_descr=ss[4].split(",")

    chars_title_ems_avg=get_avg_emb(chars_title,chars_em)
    chars_descr_ems_avg=get_avg_emb(chars_descr,chars_em)
    words_title_ems_avg=get_avg_emb(chars_title,words_em)
    words_descr_ems_avg=get_avg_emb(chars_descr,words_em)

    chars_title_emb_avg[ss[0]]=chars_title_ems_avg
    chars_descr_emb_avg[ss[0]]=chars_descr_ems_avg
    words_title_emb_avg[ss[0]]=words_title_ems_avg
    words_descr_emb_avg[ss[0]]=words_descr_ems_avg

cPickle.dump([chars_title_emb_avg,chars_descr_emb_avg,words_title_emb_avg,words_descr_emb_avg], open("quest_emb_avg.p", "wb"))
