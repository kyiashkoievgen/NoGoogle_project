import pickle
import sqlite3
import _pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from razdel import tokenize  # https://github.com/natasha/razdel
# !pip install razdel

import pymorphy2  # pip install pymorphy2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from pathlib import Path
import os
from text_prep import TextPreparation


def search_book(txt):
    pkl_vector_filename = Path(os.path.dirname(__file__), 'lib_vector.pkl')
    db_filename = Path(os.path.dirname(__file__), 'lib.db')
    hash_vector = HashingVectorizer(n_features=1000)
    tp = TextPreparation()
    text = tp.fit_transform(pd.DataFrame([txt]))
    vectorized_to_lib = hash_vector.fit_transform([text.to_string(index=False)]).toarray()

    with open(pkl_vector_filename, 'rb') as pkl_vector_file:
        clf = pickle.load(pkl_vector_file)
    pred_proba = clf.predict_proba(vectorized_to_lib)
    book_id = clf.classes_[(pred_proba > 0).ravel()]

    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()
    book_list = []

    for id_ in book_id:
        cur.execute("""SELECT userid, file_name FROM books WHERE userid=?""", (str(id_),))
        row = cur.fetchone()
        # print(id_, row)
        book_list.append(row[1])

    conn.close()
    return book_id, book_list


def search_page(book_id, txt):
    db_filename = Path(os.path.dirname(__file__), 'lib.db')
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()
    cur.execute("""SELECT userid, file_name, clf, tfidf FROM books WHERE userid=?""", (book_id,))
    row = cur.fetchone()
    conn.close()
    file_name = row[1]
    clf = _pickle.loads(row[2])
    vectorizer = _pickle.loads(row[3])
    pred = clf.predict_proba(vectorizer.transform(pd.DataFrame([txt])))
    # tp = TextPreparation()
    # prepared_txt = tp.fit_transform(pd.DataFrame([txt]))
    # vectorized_txt = norm.fit_transform(svd.fit_transform(tfidf.transform(prepared_txt)))
    # pred = clf.predict_proba(vectorized_txt)
    pages = clf.classes_[(pred > 0)[0]]
    return file_name, pages


def txt_to_html(file_name, pages, line_on_page=10):
    file_path = Path(os.path.dirname(__file__), 'inuse', file_name)
    with open(file_path, encoding="utf8") as f:
        book = f.readlines()
    for page in pages:
        last_page = page + line_on_page + line_on_page // 3
        last_page = len(book)-1 if last_page > len(book)-1 else last_page
        book[page] = f'<b><a name="{page}"></a>' + book[page]
        book[last_page] = book[last_page] + '</b>'
    return '\n'.join(book)
