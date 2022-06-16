import numpy as np
import pandas as pd

import torch
import xgboost

from sentence_transformers import SentenceTransformer, util

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class SbertClassifier:
    def __init__(self, corpus_embeddings, classifier='LogReg'):
        self.corpus_embeddings = corpus_embeddings
        self.encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        self.classifier = classifier
        if classifier == 'LogReg':
            self.clf = LogisticRegression(random_state=0, class_weight='balanced', fit_intercept=False)
        elif classifier == 'xg':
            self.clf = xgboost.XGBRegressor()
        else:
            self.clf = DecisionTreeClassifier()
        self.fitted = False

    def encode(self, text):
        text_embedding = self.encoder.encode(text, convert_to_tensor=True)
        return text_embedding

    def search(self, query):
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=32)
        hits = hits[0]
        return hits

    def fit_clf(self, X, y):
        self.clf.fit(X, y)
        self.fitted = True

    def classify_no_encode(self, X, y):
        if self.classifier != 'xg':
            return self.clf.predict(X), self.clf.predict_proba(X), self.clf.predict(X) == y
        else:
            if self.clf.predict(X) >= 0.5:
                return [1], None, None
            else:
                return [0], None, None
            # return 1, 'None', 'None' if self.clf.predict(X) >= 0.5 else 0, 'None', 'None'

    def classify(self, query, positives, negatives):
        query_embedding = self.encoder.encode(query)
        X = np.concatenate((positives, negatives))
        X = self.encoder.encode(list(X), convert_to_tensor=True)
        pos_labels = ['positive' for _ in range(len(positives))]
        neg_labels = ['negative' for _ in range(len(negatives))]
        y = np.concatenate((pos_labels, neg_labels))
        # y = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives))))
        # results = self.search(query)
        logreg = LogisticRegression(class_weight='balanced')
        logreg.fit(X, y)
        return logreg.predict(query_embedding), logreg.predict_proba(query_embedding)

    def interactive_classify(self, queries, positives, negatives, test, reruns=3):
        if reruns == 0:
            q_labels = ['positive', 'positive', 'positive', 'positive', 'positive',
                        'negative', 'negative', 'negative', 'negative', 'negative']
            print('Final run')
            prediction, probability, desc, correct = [], [], [], []
            for i, q in enumerate(test):
                pred, prob = self.classify(q, positives, negatives)
                prediction.append(pred), probability.append(prob), desc.append(q)
                correct.append(pred == q_labels[i])
            return {'prediction': prediction, 'probability': probability, 'description': desc, 'correct': correct}
        else:
            if reruns == 3:
                tested = []
                for q in test:
                    pred, _ = self.classify(q, positives, negatives)
                    tested.append(pred)
                print(tested)
            print(f'Runs remaining: {reruns}')
            clf_prediction, clf_probability = [], []

            for q in queries:
                clf_pred, clf_prob = self.classify(q, positives, negatives)
                clf_prediction.append(clf_pred)
                clf_probability.append(clf_prob[0])

            # calculate label differences
            diffs = []
            for q, pred, probs in zip(queries, clf_prediction, clf_probability):
                triplets = (q, pred, max(probs) - min(probs))
                diffs.append(triplets)
            diffs = sorted(diffs, key=lambda x: x[2])
            min_diff = diffs[:2]

            # prompt user for labels for texts in min_diff
            for val in min_diff:
                print(f'Difference in label probabilities {val[2]}')
                print(val[0])
                # print(queries[val[1]])
                label = input('Is the above text positive or negative?')
                if label == 'positive' or label == '1':
                    positives.append(val[0][0])
                elif label == 'negative' or label == '0':
                    negatives.append(val[0][0])
            # remove prompted queries from queries
            new_queries = [queries[i] for i in range(len(queries))
                           if queries[i][0] not in positives and queries[i][0] not in negatives]

            # rerun interactive_classify
            reruns -= 1
            return self.interactive_classify(new_queries, positives, negatives, test, reruns)


if __name__ == '__main__':
    df = pd.read_csv('data/cleaned_v1.csv')
    passages = df.description.tolist()
    corpus = torch.load('corpus_embeddings_bi_encoder.pt')
    sbert = SbertClassifier(corpus)
    '''hits = sbert.search('steel production')
    for hit in hits[:5]:
        print(f'{hit["score"]}   {passages[hit["corpus_id"]]}')
        print(df.iloc[hit['corpus_id']].id)'''

    train = pd.read_csv('data/train/Manufacturers.csv', delimiter=';')
    positives_id = train[(train['Rating'] == 1.0)]['Firmnav ID'].tolist()
    negatives_id = train[(train['Rating'] == 0.0) & (train['AI search'] != 'Initial')]['Firmnav ID'].tolist()
    unrated_id = train[(train['Rating'] != 1.0) & (train['Rating'] != 0.0)]['Firmnav ID'].tolist()
    positives_txt = []
    negatives_txt = []
    unrated_txt = []
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
    for fid in unrated_id:
        txt = df[df['id'] == fid].description.tolist()
        if len(txt) > 1:
            txt = [txt[0]]
        if len(txt) > 0:
            unrated_txt.append(txt)

    positives_txt = [item for sublist in positives_txt for item in sublist]
    negatives_txt = [item for sublist in negatives_txt for item in sublist]
    positives_txt = positives_txt[5:]
    negatives_txt = negatives_txt[5:]
    qs = positives_txt[:5] + negatives_txt[:5]
    qs = [[val] for val in qs]
    interactive_results = sbert.interactive_classify(unrated_txt, positives_txt, negatives_txt, qs, reruns=0)
    interactive_results = pd.DataFrame(interactive_results)
    interactive_results.to_csv('interactive_results.csv', index=True)
    prediction, probability, desc = [], [], []
    for txt in unrated_txt:
        pred, proba = sbert.classify(txt, positives_txt, negatives_txt)
        prediction.append(pred)
        probability.append(proba)
        desc.append(txt[0])
    result = {'prediction': prediction, 'probability': probability, 'description': desc}
    result = pd.DataFrame(result)
    # result.to_csv('result.csv', index=True)
    print(result)
    print(result[result['prediction'] == 'positive'].shape[0])
    print(result[result['prediction'] == 'positive'].description)
