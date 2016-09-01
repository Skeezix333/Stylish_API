import os
import pandas as pd
import numpy as np
from collections import Counter,defaultdict

PATH = '../sample_proj'

def listdir_nohidden(path):
    not_hidden = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            not_hidden.append(f)
    return not_hidden

authorlist = listdir_nohidden(PATH)

authordict ={}

for author in authorlist:
    booklist = listdir_nohidden('{}/{}'.format(PATH, author))
    bookdict = {}
    for book in booklist:
        txt_list = listdir_nohidden('{}/{}/{}'.format(PATH, author, book))
        temp_dict = {}
        for txt in txt_list:
            if txt.endswith(".txt"):
                temp_dict= open('{}/{}/{}/{}'.format(PATH,author, book, txt), 'r').read()
                bookdict[book] = temp_dict
    authordict[author] = bookdict


final_dict = defaultdict(list)
for author, books in authordict.iteritems():
    for book, txt in books.iteritems():
        final_dict['Author'].append(author)
        final_dict['Book_Title'].append(book)
        final_dict['Book_Text'].append(txt)

df = pd.DataFrame.from_dict(final_dict)

def split_by_n( seq, n ):
    """A generator to divide a sequence into chunks of n units."""
    while seq:
        yield seq[:n]
        seq = seq[n:]

split_txt = []
for txt in df['Book_Text']:
    split_txt.append(list(split_by_n(txt.split(), 10000)))

df['Split_Text']= split_txt
df = df.drop('Book_Text', axis=1)

rows = []
_ = df.apply(lambda row: [rows.append([row['Author'], row['Book_Title'], nn])
                         for nn in row.Split_Text], axis=1)
df_new = pd.DataFrame(rows, columns=df.columns)

def convertToString(a, s):
    la = len(a)
    b = a[0:la] # copy list (like slice() in JavaScript) — "buffer"
    for i in xrange(0, la): # iterate
        b[i] = str(b[i]) # convert each to string
    return s.join(b) # return all string

string_list=[]
for txt in df.Split_Text:
    for t in txt:
        string_list.append(convertToString(t, " "))

df_new.Split_Text=string_list

print df_new
