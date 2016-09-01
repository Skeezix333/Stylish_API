import os
import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import sklearn
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, stem
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
import string
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import cPickle
import pickle

def split_by_n( seq, n ):
    """A generator to divide a sequence into chunks of n units."""
    while seq:
        yield seq[:n]
        seq = seq[n:]

def convertToString(a, s):
    la = len(a)
    b = a[0:la]
    for i in xrange(0, la):
        b[i] = str(b[i])
    return s.join(b)

def clean_text(text):
    no_cap =' '.join(word for word in text.split() if word.islower())
    exclude = set(string.punctuation)
    no_punc = ''.join(char for char in no_cap if char not in exclude)
    return no_punc

def clean_up(txt):
    split_txt = []
    print txt
    split_txt.append(list(split_by_n(txt.split(), 5000)))

    df_new = pd.DataFrame({'Split_Text':split_txt})

    string_list=[]
    for txt in df_new.Split_Text:
        for t in txt:
            string_list.append(convertToString(t, " "))

    df_new = pd.DataFrame({'Split_Text':string_list})


    avg_sent_len_list = []
    for x in df_new.Split_Text:
        length = np.mean([len(s.split(' ')) for s in sent_tokenize(x)])
        avg_sent_len_list.append(length)

    df_new['Avg_Sen'] = avg_sent_len_list



    comma_list = []
    semic_list = []
    quote_list = []
    excl_list = []
    question_list = []
    and_list = []
    if_list = []
    but_list = []
    must_list = []
    might_list = []
    more_list = []
    that_list = []
    this_list = []
    very_list = []
    however_list =[]



    for x in df_new.Split_Text:
        temp_com = x.count(',')
        temp_semic = x.count(';')
        temp_quote = x.count('"')
        temp_excl = x.count('!')
        temp_question = x.count('?')
        temp_and = x.lower().count('and')
        temp_if = x.lower().count('if')
        temp_but = x.lower().count('but')
        temp_must= x.lower().count('must')
        temp_might = x.lower().count('might')
        temp_more = x.lower().count('more')
        temp_that = x.lower().count('that')
        temp_this = x.lower().count('this')
        temp_very = x.lower().count('very')
        temp_however = x.lower().count('however')
        comma_list.append(temp_com)
        semic_list.append(temp_semic)
        quote_list.append(temp_quote)
        excl_list.append(temp_excl)
        question_list.append(temp_question)
        and_list.append(temp_and)
        if_list.append(temp_if)
        but_list.append(temp_but)
        must_list.append(temp_must)
        might_list.append(temp_might)
        more_list.append(temp_more)
        that_list.append(temp_that)
        this_list.append(temp_this)
        very_list.append(temp_very)
        however_list.append(temp_however)

    df_new['Comma'] = comma_list
    df_new['SemiC'] = semic_list
    df_new['Quote'] = quote_list
    df_new['Excl'] = excl_list
    df_new['Question'] = question_list
    df_new['And'] = and_list
    df_new['If'] = if_list
    df_new['But'] = but_list
    df_new['Must'] = must_list
    df_new['Might'] = might_list
    df_new['More'] = more_list
    df_new['That'] = that_list
    df_new['This'] = this_list
    df_new['Very'] = very_list
    df_new['However'] = however_list

    string_list_newer=[]
    for txt in df_new.Split_Text:
            string_list_newer.append(clean_text(txt))

    df_newer = df_new

    df_newer.Split_Text = string_list_newer

    ttr_list = []
    for x in df_newer.Split_Text:
        temp_ttr = float(len(set(x.split(' '))))/(len(x.split(' ')))
        ttr_list.append(temp_ttr)

    df_newer['TTR']=ttr_list

    stemmer = SnowballStemmer('english')
    stemmed_string_list = []
    for x in df_newer.Split_Text:
        temp_stem_list = []
        for word in x.split(' '):
            temp_stem_list.append(stemmer.stem(word))
        stemmed_string_list.append(' '.join(temp_stem_list))

    df_newer['Stemmed_Text'] = stemmed_string_list

    stem_dttr_list = []
    for x in df_newer.Stemmed_Text:
        temp_stem_dttr = float(len(set(x.split(' '))))/(len(x.split(' ')))
        stem_dttr_list.append(temp_stem_dttr)

    df_newer['Stem_TTR'] = stem_dttr_list

    # lemma = WordNetLemmatizer()
    # lemma_string_list = []
    # for x in df_newer.Split_Text:
    #     temp_lemma_list = []
    #     for word in x.split(' '):
    #         temp_lemma_list.append(lemma.lemmatize(word))
    #     lemma_string_list.append(' '.join(temp_lemma_list))
    #
    # df_newer['Lemma_Text'] = lemma_string_list
    #
    #
    # lemma_dttr_list = []
    # for x in df_newer.Lemma_Text:
    #     temp_lemma_dttr = float(len(set(x.split(' '))))/(len(x.split(' ')))
    #     lemma_dttr_list.append(temp_lemma_dttr)
    #
    # df_newer['Lemma_TTR'] = lemma_dttr_list

    avg_word_len_list = []
    for x in df_newer.Split_Text:
        avg_word_len_list.append(len(''.join(x.split(' ')))/float((len(x.split()))))

    df_newer['Avg_Word'] = avg_word_len_list

    avg_word_stop_list = []
    stop = stopwords.words('english')
    for x in df_newer.Split_Text:
        no_stop_list = [i for i in x.split(' ') if i not in stop]
        avg_word_stop_list.append(len(''.join(no_stop_list))/float((len(no_stop_list))))

    df_newer['Avg_NoStop_Word'] = avg_word_stop_list

    # word_stop_list = []
    # stop = stopwords.words('english')
    # for x in df_newer.Split_Text:
    #     no_stop_list = [i for i in x.split(' ') if i not in stop]
    #     word_stop_list.append(' '.join(no_stop_list))
    #
    # df_newer['Stopless_Text'] = word_stop_list

    count_vect = TfidfVectorizer(stop_words='english', vocabulary = pickle.load(open("../training_vocab.pkl", "rb")))
    X_train_counts = count_vect.fit_transform(df_newer.Stemmed_Text)

    pca = pickle.load(open("../PCA.pkl", "rb"))


    X_centered = sklearn.preprocessing.scale(X_train_counts.toarray())
    X_pca = pca.transform(X_centered)

    PCA_Columns = ['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10']
    df_PCA = pd.DataFrame(X_pca, columns = PCA_Columns)


    df_final = pd.concat([df_newer, df_PCA], axis=1)

    df_final.drop(['Split_Text','Stemmed_Text'], axis=1, inplace=True)

    return df_final
