VERSION_STR = 'v1.0.0'

authorlist_first = ['Adams, Douglas','Amis, Martin','Atwood, Margaret','Austen, Jane','Banks, Iain','Carre, John Le',\
'Chandler, Raymond','Conrad, Joseph','Dickens, Charles','Dumas, Alexandre','Eliot, George','Ellis, Bret Easton',\
'Fielding, Henry','Forster, E. M_','Hardy, Thomas','Hemingway, Ernest','Hugo, Victor','Ishiguro, Kazuo',\
'James, Henry','Lawrence, D. H_','McEwan, Ian','Murakami, Haruki','Nabokov, Vladimir','Orwell, George',\
'Proust, Marcel','Pynchon, Thomas','Roth, Philip','Scott, Sir Walter','Stevenson, Robert Louis',\
'Tolkien, J. R. R_','Trollope, Anthony','Turgenev, Ivan','Vonnegut, Kurt','Wells, H. G_','Woolf, Virginia']

import sys
import requests
import numpy as np
import json
from error import Error
from flask import Blueprint, request, jsonify
from random import randint, uniform
# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from clean_txt import clean_up
import cPickle
import pickle

sys.path.append('../')

# run build model on a weekly basis so that patents can be stored
# in a compressed format (this part is the time consuming part)
# After this, run use_model to unload pickles/msgpacks and run
# scoring function to return the appropriate object.

#import use_model

#df, abstracts_tfidf, tfidf = use_model.unpickle()
# user_text -- how to generate/store this?
 #-- think about how this may be implemented (NOT NECESSARY AT THE MOMENT)



blueprint = Blueprint(VERSION_STR, __name__)
@blueprint.route('/return_author_probability', methods=['POST'])
def return_author_probability():
    '''
    Use this endpoint to give you a list of author styles most similar to the user generated text.
    ---
    tags:
     - v1.0.0
    responses:
     200:
       description: Returns a dictionary with 5 authors and their associated propabilities in relation to the user's text.
     default:
       description: Unexpected error
       schema:
         $ref: '#/definitions/Error'
    parameters:
     - name: txt
       in: query
       description: User generated text
       required: false
       type: string
     - name: txt_file
       in: formData
       description: User provided txt file
       required: false
       type: file
    consumes:
     - multipart/form-data
     - application/x-www-form-urlencoded
    '''
    if 'txt' in request.args:
        user_text = request.args['txt']
    elif 'txt_file' in request.files:
        temp_text = request.files['txt_file']
        user_text = temp_text.read()


    df = clean_up(user_text)

    with open('../RFmodel.cpickle', 'rb') as f:
        rf = cPickle.load(f)


    read_dictionary = np.load('../author_id.npy').item()
    inv_read_dictionary = {v: k for k, v in read_dictionary.items()}

    preds = rf.predict_proba(df)
    preds_sum = np.sum(preds, axis=0)
    top_authors = np.argsort(preds_sum)[::-1][0:5]
    top_preds = preds_sum[top_authors]
    top_preds = top_preds / preds.shape[0]
    top_preds = top_preds / sum(top_preds)

    author_name = []
    temp_first = []
    temp_last = []

    for name in top_authors:
        author_name.append(inv_read_dictionary[name])

    for name in author_name:
        name_split = name.split(', ')
        temp_first.append(name_split[1])
        temp_last.append(name_split[0])

    # temp_auth_list = []
    # temp_prob_list = []
    # temp_first = []
    # temp_last = []
    #
    #
    #
    # for i in range(5):
    #     temp_prob_list.append(round(uniform(0, 1), 3))
    # temp_auth_list.extend(np.random.choice(authorlist_first, size = 5, replace = False))
    #
    # for name in temp_auth_list:
    #     name_split = name.split(', ')
    #     temp_first.append(name_split[1])
    #     temp_last.append(name_split[0])
    #
    #
    results = [{'first_name': auth_first, 'last_name': auth_last, 'prob': prob} for auth_first, auth_last, prob in zip(temp_first, temp_last, top_preds)]

    d = {'results' : results}
    response = jsonify(d)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response




from app import app
app.register_blueprint(blueprint, url_prefix='/'+VERSION_STR)
