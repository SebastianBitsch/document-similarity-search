import numpy as np
import pandas as pd

import torch

from sentence_transformers import SentenceTransformer, util

from sklearn.linear_model import LogisticRegression


class SbertClassifier:
    def __init__(self, corpus_embeddings):
        self.corpus_embeddings = corpus_embeddings
        self.encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    def search(self, query):
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=32)
        hits = hits[0]
        return hits

    def classify(self, query, positives, negatives):
        X = np.concatenate((positives, negatives))
        X = self.encoder.encode(list(X), convert_to_tensor=True)
        y = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives))))
        # results = self.search(query)
        logreg = LogisticRegression()
        logreg.fit(X, y)
        return logreg.predict(self.encoder.encode(query))


if __name__ == '__main__':
    df = pd.read_csv('data/cleaned_v1.csv')
    passages = df.description.tolist()
    corpus = torch.load('corpus_embeddings_bi_encoder.pt')
    sbert = SbertClassifier(corpus)
    '''hits = sbert.search('steel production')
    for hit in hits[:5]:
        print(f'{hit["score"]}   {passages[hit["corpus_id"]]}')
        print(df.iloc[hit['corpus_id']].id)'''

    train = pd.read_csv('data/train/Healthcare.csv', delimiter=';')
    positives_id = train[train['Rating'] == 1.0]['Firmnav ID'].tolist()
    negatives_id = train[train['Rating'] == 0.0]['Firmnav ID'].tolist()
    positives_txt = []
    negatives_txt = []
    for fid in positives_id:
        txt = df[df['id'] == fid].description.tolist()
        if len(txt) > 1:
            txt = [txt[0]]
        if len(txt) > 0:
            positives_txt.append(txt)
    for fid in negatives_id:
        txt = df[df['id'] == fid].description.tolist()
        if len(txt) > 1:
            txt = [txt[0]]
        if len(txt) > 0:
            negatives_txt.append(txt)

    positives_txt = [item for sublist in positives_txt for item in sublist]
    negatives_txt = [item for sublist in negatives_txt for item in sublist]
    q = df[df['id'] == 'SE5567738249'].description.tolist()
    print(sbert.classify(q, positives_txt, negatives_txt))
    print('niggers')


