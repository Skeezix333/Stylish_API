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

df_final= pd.read_pickle('big_df_final_RF.pkl')

y_final = df_final['Author']

collist = df_final.columns.tolist()
collist.remove('Author')
collist.remove('Book_Title')

X_final = df_final[collist]

#X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X_final, y_final)

rf = RandomForestClassifier(random_state=42, min_samples_split=3)
rf.fit(X_final, y_final)

with open('RFmodel.cpickle', 'wb') as f:
    cPickle.dump(rf, f)

# with open('RFmodel.cpickle', 'rb') as f:
#     rf = cPickle.load(f)
#
#
# preds = rf.predict(new_X)
#
# read_dictionary = np.load('author_id.npy').item()
