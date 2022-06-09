from sbert_classifier import SbertClassifier

import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut


trainfiles = ['Computer vision.csv', 'Consulting.csv', 'Fintech.csv', 'Fish processing equipment.csv', 'Healthcare.csv',
              'House builders.csv', 'Industrial vertical investor.csv', 'Innovative.csv', 'IoT.csv', 'IT freelance.csv',
              'M&A advisors.csv', 'Manufacturers.csv', 'Online games.csv', 'Payments tech.csv', 'PE fund.csv',
              'Procurement software.csv', 'Resource-efficiency.csv', 'Sustainability.csv', 'SaaS.csv',
              'Wind turbine tech.csv', ]

corpus = torch.load('corpus_embeddings_bi_encoder.pt')
clf = SbertClassifier(corpus)
np.random.seed(40)
data = pd.read_csv('data/cleaned_v1.csv')


'''def LeaveOneOut(sbert=clf, filenames=None):
    if filenames is None:
        filenames = trainfiles
    # left_out = np.random.choice(filenames)
    # filenames.remove(left_out)
    num_correct = 0
    num_false = 0
    total = 0
    for name in filenames:
        df_train = pd.read_csv(f'data/train/{name}', delimiter=';')
        positive_ids = df_train[df_train['Rating'] == 1.0]['Firmnav ID'].tolist()
        negative_ids = df_train[df_train['Rating'] == 0.0]['Firmnav ID'].tolist()
        positive_txt = []
        negative_txt = []

        for fid in positive_ids:
            txt = data[data['id'] == fid].description.tolist()
            if len(txt) > 1:
                txt = [txt[0]]
            if len(txt) > 0:
                positive_txt.append(txt)
        for fid in negative_ids:
            txt = data[data['id'] == fid].description.tolist()
            if len(txt) > 1:
                txt = [txt[0]]
            if len(txt) > 0:
                negative_txt.append(txt)

        positive_txt = [item for sublist in positive_txt for item in sublist]
        negative_txt = [item for sublist in negative_txt for item in sublist]
        pos_labels = ['positive' for _ in range(len(positive_txt))]
        neg_labels = ['negative' for _ in range(len(negative_txt))]
        X = np.concatenate((positive_txt, negative_txt))
        X = sbert.encode(list(X))
        y = np.concatenate((pos_labels, neg_labels))
        for i in range(len(X)):
            Xtest = X[i]
            ytest = y[i]
            Xtrain = [x for j, x in enumerate(X) if j != i]
            ytrain = [x for j, x in enumerate(y) if j != i]
            sbert.fit_logreg(Xtrain, ytrain)
            prediction, probabilities, correct = sbert.classify_no_encode(Xtest, ytest)
            if correct:
                num_correct += 1
            else:
                num_false += 1
            total += 1
    return num_correct, num_false, total'''


if __name__ == '__main__':
    print(LeaveOneOut())
