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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

df_newer = pd.read_pickle('big_df_newer.pkl')

count_vect = TfidfVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(df_newer.Stemmed_Text)

pickle.dump(count_vect.vocabulary_,open("training_vocab.pkl","wb"))

#how to read back in
#pickle.load(open("training_vocab.pkl", "rb"))


print '1'

pca = PCA(n_components=10)

print '2'

X_centered = sklearn.preprocessing.scale(X_train_counts.toarray())
X_pca = pca.fit_transform(X_centered)

pickle.dump(pca,open("PCA.pkl","wb"))
#how to read back in
#pickle.load(open("PCA.pkl", "rb"))


print '3'

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set2(y[i] / 50.), fontdict={'weight': 'bold', 'size': 12})

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,1.1])

    if title is not None:
        plt.title(title, fontsize=16)

print '4'

PCA_Columns = ['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10']
df_PCA = pd.DataFrame(X_pca, columns = PCA_Columns)

print '5'

df_final = pd.concat([df_newer, df_PCA], axis=1)

df_final.drop(['Split_Text','Stemmed_Text'], axis=1, inplace=True)

# y_final = df_final['Author']
#
# collist = df_final.columns.tolist()
# collist.remove('Author')
# collist.remove('Book_Title')
#
# X_final = df_final[collist]
#
# X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X_final, y_final, train_size = 1.0)
#
# model = TSNE(n_components=2, random_state=0)
#
# X_centered_final = sklearn.preprocessing.scale(X_train_fin)
# X_tsne = model.fit_transform(X_centered_final)
#
# plot_embedding(X_tsne, y_train_fin.tolist())
#
# plt.savefig('big_tsne.png')


df_final.to_pickle('big_df_final_RF.pkl')
