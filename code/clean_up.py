import numpy as np
import pandas as pd
import os
import string


def clean_file(filename):
    open_file = open(filename, 'r').read()
    no_cap =' '.join(word for word in open_file.split() if word.islower())
    exclude = set(string.punctuation)
    no_punc = ''.join(char for char in no_cap if char not in exclude)
    return no_punc

def clean_text(text):
    no_cap =' '.join(word for word in text.split() if word.islower())
    exclude = set(string.punctuation)
    no_punc = ''.join(char for char in no_cap if char not in exclude)
    return no_punc

if __name__ == '__main__':
    s = clean_file('sample_proj/Dostoyevsky, Fyodor/Crime and Punishment/Crime and Punishment - Fyodor Dostoyevsky.txt')
    print s
