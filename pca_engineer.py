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

df_newer = pd.read_pickle('first_df_newer.pkl')

count_vect = TfidfVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(df_newer.Stemmed_Text)

pca = PCA(n_components=10)

X_centered = sklearn.preprocessing.scale(X_train_counts.toarray())
X_pca = pca.fit_transform(X_centered)

PCA_Columns = ['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10']
df_PCA = pd.DataFrame(X_pca, columns = PCA_Columns)

df_final = pd.concat([df_newer, df_PCA], axis=1)

df_final.to_pickle('first_df_final.pkl')
