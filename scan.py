from text_prep import create_doc_model, create_classification

import os
import sys
import pickle
import _pickle
import sqlite3
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path


def scan_input():
    pkl_vector_filename = Path(os.path.dirname(__file__), 'lib_vector.pkl')
    input_dir = Path(os.path.dirname(__file__), 'Input')
    inuse_dir = Path(os.path.dirname(__file__), 'inuse')

    conn = sqlite3.connect(r'lib.db')
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS books(
       userid INTEGER PRIMARY KEY,
       file_name TEXT,
       clf BLOB,
       tfidf BLOB,
       vectorized_to_lib BLOB);
    """)
    conn.commit()

    txt_files = [name for name in os.listdir(input_dir) if name.endswith('.txt')]

    x = pd.DataFrame()
    y = pd.DataFrame()
    # ['Iskushieniie budushchim - Nieizviestnyi.txt']:  #
    for file_name in txt_files:
        # print(file_name)
        with open(Path(input_dir, file_name), encoding="utf8") as f:
            book = f.readlines()

            # try:
            #     with open(pkl_vector_filename, 'rb') as pkl_vector_file:
            #         lib_vector = pickle.load(pkl_vector_file)
            #         clf, tfidf, lib_vector, vectorized_to_lib = create_doc_model(book, lib_vector)
            #
            # except FileNotFoundError:
            #     clf, tfidf, lib_vector, vectorized_to_lib = create_doc_model(book)
            #
            # with open(pkl_vector_filename, 'wb') as pkl_vector_file:
            #     pickle.dump(lib_vector, pkl_vector_file)
            clf, tfidf, vectorized_to_lib = create_doc_model(book)
            pickled_clf = _pickle.dumps(clf)
            pickled_tfidf = _pickle.dumps(tfidf)
            pickled_vectorized_to_lib = _pickle.dumps(vectorized_to_lib)
            cur.execute("INSERT INTO books (file_name, clf, tfidf, vectorized_to_lib) VALUES( ?, ?, ?, ?);",
                        (file_name, pickled_clf, pickled_tfidf, pickled_vectorized_to_lib))
            conn.commit()
            y = pd.concat([y, pd.DataFrame([{'book_id': cur.lastrowid}])])
            x = pd.concat([x, pd.DataFrame(vectorized_to_lib)])
        # with open('inuse/' + file_name, '+w', encoding="utf8") as f:
        #     f.write(str(book))
        os.replace(Path(input_dir, file_name), Path(inuse_dir, file_name))
        # os.remove('input/' + file_name)
    # clf = KNeighborsClassifier(n_neighbors=10)
    # clf.fit(x, y.to_numpy().ravel())
    # with open(pkl_vector_filename, 'wb') as pkl_vector_file:
    #     pickle.dump(clf, pkl_vector_file)
    create_classification(10)


if __name__ == "__main__":
    scan_input()
