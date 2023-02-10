# предобработка текстов
import _pickle
import os
import pickle
import re
import sqlite3

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse.dia import dia_matrix
from sklearn.feature_extraction.text import HashingVectorizer

from razdel import tokenize

import pymorphy2

from sklearn.feature_extraction.text import TfidfVectorizer
import pathlib
from pathlib import Path


def clean_text(text):
    '''
    очистка текста

    на выходе очищеный текст

    '''
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.strip('\n').strip('\r').strip('\t')
    text = re.sub("-\s\r\n\|-\s\r\n|\r\n", '', str(text))

    text = re.sub("[0-9]|[-—.,:;_%©«»?*!@#№$^•·&()]|[+=]|[[]|[]]|[/]|", '', text)
    text = re.sub(r"\r\n\t|\n|\\s|\r\t|\\n", ' ', text)
    text = re.sub(r'[\xad]|[\s+]', ' ', text.strip())

    # tokens = list(tokenize(text))
    # words = [_.text for _ in tokens]
    # words = [w for w in words if w not in stopword_ru]

    # return " ".join(words)
    return text


class TextPreparation(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.cache = {}
        self.stopword_ru = stopwords.words('russian')
        self.morph = pymorphy2.MorphAnalyzer()
        path = Path(os.path.dirname(__file__), 'stopwords.txt')
        with open(path, encoding="utf8") as f:
            additional_stopwords = [w.strip() for w in f.readlines() if w]
        self.stopword_ru += additional_stopwords

    def lemmatization(self, text):
        '''
        лемматизация
            [0] если зашел тип не `str` делаем его `str`
            [1] токенизация предложения через razdel
            [2] проверка есть ли в начале слова '-'
            [3] проверка токена с одного символа
            [4] проверка есть ли данное слово в кэше
            [5] лемматизация слова
            [6] проверка на стоп-слова

        на выходе лист отлемматизированых токенов
        '''

        # [0]
        if not isinstance(text, str):
            text = str(text)

        # [1]
        tokens = list(tokenize(text))
        words = [_.text for _ in tokens]

        words_lem = []
        for w in words:
            if w[0] == '-':  # [2]
                w = w[1:]
            if len(w) > 1:  # [3]
                if w in self.cache:  # [4]
                    words_lem.append(self.cache[w])
                else:  # [5]
                    temp_cach = self.cache[w] = self.morph.parse(w)[0].normal_form
                    words_lem.append(temp_cach)

        words_lem_without_stopwords = [i for i in words_lem if not i in self.stopword_ru]  # [6]

        return ' '.join(words_lem_without_stopwords)

    def fit(self, txt, y=None):
        return self

    def transform(self, txt, y=None):
        txt = txt.apply(lambda x: clean_text(x), 1)
        txt = txt.apply(lambda x: self.lemmatization(x), 1)
        return txt


class TfidfVectorizerPartal(TfidfVectorizer):

    # def __init__(self):
    #     super().__init__(self)
    #     self.n_docs = 0

    def partial_fit(self, X):
        max_idx = max(self.vocabulary_.values())

        print(self._check_n_features(X, reset=True))
        for a in X:
            # update vocabulary_
            if self.lowercase: a = a.lower()
            tokens = re.findall(self.token_pattern, a)
            for w in tokens:
                if w not in self.vocabulary_:
                    max_idx += 1
                    self.vocabulary_[w] = max_idx

            # update idf_
            df = (self.n_docs + self.smooth_idf) / np.exp(self.idf_ - 1) - self.smooth_idf
            self.n_docs += 1
            df.resize(len(self.vocabulary_))
            for w in tokens:
                df[self.vocabulary_[w]] += 1
            idf = np.log((self.n_docs + self.smooth_idf) / (df + self.smooth_idf)) + 1
            self._tfidf._idf_diag = dia_matrix((idf, 0), shape=(len(idf), len(idf)))


def create_doc_model(txt, line_on_page=10, neighbors_book=10):
    book_df = pd.DataFrame(columns=['Page', 'txt'])
    tp = TextPreparation()
    # svd = TruncatedSVD(n_components=100)
    # norm = Normalizer(copy=False)
    # tfidf = TfidfVectorizer()
    clf = KNeighborsClassifier(n_neighbors=neighbors_book)
    for i in range(0, len(txt), line_on_page):
        text = txt[i:i + line_on_page + line_on_page // 3]
        print(text, '\n', '\n')
        book_df = pd.concat([book_df, pd.DataFrame([{'Page': i, 'txt': text}])])
    book_df['Page'] = book_df['Page'].astype(int)
    vectorizer = make_pipeline(TextPreparation(), TfidfVectorizer(max_df=0.5, min_df=1),
                               TruncatedSVD(n_components=100), Normalizer(copy=False))
    prepared_txt = tp.fit_transform(book_df['txt'])
    txt_aad = prepared_txt.to_string(index=False)
    # vectorized_txt = norm.fit_transform(svd.fit_transform(tfidf.fit_transform(prepared_txt)))
    clf.fit(vectorizer.fit_transform(book_df['txt']), book_df['Page'])
    add_vector = HashingVectorizer(n_features=1000)
    vectorized_to_lib = add_vector.fit_transform([txt_aad]).toarray()
    # if add_vector is None:
    #     add_vector = HashingVectorizer(n_features=1000)#TfidfVectorizerPartal()
    #     vectorized_to_lib = add_vector.fit_transform([txt_aad]).toarray()
    #     #add_vector.n_docs = 1
    #     #print(len(add_vector.get_feature_names_out()))
    # else:
    #     add_vector.partial_fit([txt_aad])
    #     #print(len(add_vector.get_feature_names_out()))
    #     vectorized_to_lib = add_vector.transform([txt_aad]).toarray()
    # print(prepared_txt.to_string(index=False))
    # for i in ff[0]:
    #     print(i, '\n')
    # vectorized_to_lib = norm.fit_transform(svd.fit_transform(vectorized_to_lib))
    # print(vectorized_to_lib)
    return clf, vectorizer, vectorized_to_lib
    # add_vector, vectorized_to_lib
    # .to_string(index=False)


def create_classification(neighbors=10):
    pkl_vector_filename = Path(os.path.dirname(__file__), 'lib_vector.pkl')
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    conn = sqlite3.connect(r'lib.db')
    cur = conn.cursor()
    cur.execute("""SELECT * FROM books""")
    records = cur.fetchall()
    x = pd.DataFrame()
    y = pd.DataFrame()
    for row in records:
        y = pd.concat([y, pd.DataFrame([{'book_id': row[0]}])])
        unpickled = _pickle.loads(row[4])
        x = pd.concat([x, pd.DataFrame(unpickled)])

    clf.fit(x, y.to_numpy().ravel())
    with open(pkl_vector_filename, 'wb') as pkl_vector_file:
        pickle.dump(clf, pkl_vector_file)
