from sbert_classifier import SbertClassifier

import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import PrecisionRecallDisplay

from tqdm import tqdm
import matplotlib.pyplot as plt


trainfiles = ['Computer vision.csv', 'Consulting.csv', 'Fintech.csv', 'Fish processing equipment.csv', 'Healthcare.csv',
              'House builders.csv', 'Industrial vertical investor.csv', 'Innovative.csv', 'IoT.csv', 'IT freelance.csv',
              'M&A advisors.csv', 'Manufacturers.csv', 'Online games.csv', 'Payments tech.csv', 'PE fund.csv',
              'Procurement software.csv', 'Resource-efficiency.csv', 'Sustainability.csv', 'SaaS.csv',
              'Wind turbine tech.csv', ]

corpus = torch.load('corpus_embeddings_bi_encoder.pt')
clf = SbertClassifier(corpus)
data = pd.read_csv('data/cleaned_v1.csv')


def get_results(sbert=clf, filenames=None):
    if filenames is None:
        filenames = trainfiles
    # left_out = np.random.choice(filenames)
    # filenames.remove(left_out)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = 0
    y_true, y_pred = [], []
    for name in filenames:
        np.random.seed(40)
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
        loo = LeaveOneOut()
        loo.get_n_splits(X)

        for train_index, test_index in tqdm(loo.split(X)):
            # print("TRAIN:", train_index, "TEST:", test_index)
            Xtrain, Xtest = X[train_index], X[test_index]
            ytrain, ytest = y[train_index], y[test_index]
            # print(Xtrain, Xtest, ytrain, ytest)
            sbert.fit_logreg(Xtrain, ytrain)
            prediction, _, correct = sbert.classify_no_encode(Xtest, ytest)
            if prediction[0] == 'positive' and correct:
                true_positive += 1
            elif prediction[0] == 'negative' and correct:
                true_negative += 1
            elif prediction[0] == 'positive' and not correct:
                false_positive += 1
            elif prediction[0] == 'negative' and not correct:
                false_negative += 1
            y_true.append(ytest[0])
            y_pred.append(prediction[0])
            total += 1
    return true_negative, true_positive, false_positive, false_negative, total, y_true, y_pred


if __name__ == '__main__':
    recalls = []
    precisions = []
    Cs, Fs = [], []
    for _ in tqdm(range(10)):
        TN, TP, FP, FN, total, y_true, y_pred = get_results()
        y_true = [1 if val == 'positive' else 0 for val in y_true]
        y_pred = [1 if val == 'positive' else 0 for val in y_pred]
        '''disp = PrecisionRecallDisplay.from_predictions(np.array(y_true), np.array(y_pred))
        disp.plot()
        plt.show()'''
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        '''print(f'The number of True Positives were {TP}\n'
              f'The number of True Negatives were {TN}\n'
              f'The total number of correct were {TP+TN}\n'
              f'The number of False Positives were {FP}\n'
              f'The number of False Negatives were {FN}\n'
              f'The ratio of correct predictions were {(TP+TN)/total}\n'
              f'The ratio of false predictions were {(FP+FN)/total}\n'
              f'Precision is {precision}\n'
              f'Recall is {recall}')'''
        precisions.append(precision)
        recalls.append(recall)
        Cs.append(TP+TN)
        Fs.append(FP+FN)

    mean_precisions = np.mean(precisions)
    mean_recalls = np.mean(recalls)
    mean_correct = np.mean(Cs)
    mean_false = np.mean(Fs)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(mean_precisions)
    ax2.plot(mean_recalls)
    ax3.plot(mean_correct)
    ax4.plot(mean_false)
    plt.show()


