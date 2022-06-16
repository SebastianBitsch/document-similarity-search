from sbert_leave_one_out import get_results
import json
from statsmodels.stats.contingency_tables import mcnemar

f = open('mcnemar_tfidf.json')
mcnemarstf = json.load(f)
f.close()
mcnemarstf = [1 if val[i] == 'True' else 0 for val in mcnemarstf.values() for i in range(len(val))]
f = open('mcnemar_custom.json')
mcnemarsfeatures = json.load(f)
f.close()
mcnemarsfeatures = [1 if val[i] == '1' else 0 for val in mcnemarsfeatures.values() for i in range(len(val))]
_, _, _, _, _, y_true, y_pred, _, _, y_ids = get_results()
#y_true = [1 if val == 'positive' else 0 for val in y_true]
#y_pred = [1 if val == 'negative' else 0 for val in y_pred]
mcnemarsbert = [v1 == v2 for v1, v2 in zip(y_true, y_pred)]
f = open('xgboost_mcnemar_custom.json')
xgboostFeatures = json.load(f)
f.close()
xgboostFeatures = [1 if val[i] == '1' else 0 for val in xgboostFeatures.values() for i in range(len(val))]
f = open('xgboost_mcnemar_tfidf.json')
xgboostTF = json.load(f)
f.close()
xgboostTF = [1 if val[i] == '1' else 0 for val in xgboostTF.values() for i in range(len(val))]
f = open('xgboost_mcnemar_tfidf_names.json')
xgboost_names = json.load(f)
xgboost_names = [val[i] for val in xgboost_names.values() for i in range(len(val))]
print(xgboost_names, y_ids)

n11, n12, n21, n22 = 0, 0, 0, 0
for sb, tf in zip(mcnemarsbert, mcnemarstf):
    if sb == 1 and tf == 1:
        n11 += 1
    elif sb == 1 and tf == 0:
        n12 += 1
    elif sb == 0 and tf == 1:
        n21 += 1
    elif sb == 0 and tf == 0:
        n22 += 1

data = [[n11, n12], [n21, n22]]
print('sbert and tfidf', mcnemar(data, exact=False, correction=True))

n11, n12, n21, n22 = 0, 0, 0, 0
for sb, tf in zip(mcnemarsbert, mcnemarsfeatures):
    if sb == 1 and tf == 1:
        n11 += 1
    elif sb == 1 and tf == 0:
        n12 += 1
    elif sb == 0 and tf == 1:
        n21 += 1
    elif sb == 0 and tf == 0:
        n22 += 1

data = [[n11, n12], [n21, n22]]
print('sbert and features', mcnemar(data, exact=False, correction=True))

n11, n12, n21, n22 = 0, 0, 0, 0
for sb, tf in zip(mcnemarstf, mcnemarsfeatures):
    if sb == 1 and tf == 1:
        n11 += 1
    elif sb == 1 and tf == 0:
        n12 += 1
    elif sb == 0 and tf == 1:
        n21 += 1
    elif sb == 0 and tf == 0:
        n22 += 1

data = [[n11, n12], [n21, n22]]
print('tfidf and features', mcnemar(data, exact=False, correction=True))

n11, n12, n21, n22 = 0, 0, 0, 0
for sb, tf in zip(mcnemarsbert, xgboostTF):
    if sb == 1 and tf == 1:
        n11 += 1
    elif sb == 1 and tf == 0:
        n12 += 1
    elif sb == 0 and tf == 1:
        n21 += 1
    elif sb == 0 and tf == 0:
        n22 += 1

data = [[n11, n12], [n21, n22]]
print('sbert and xgboost tfidf', mcnemar(data, exact=False, correction=True))

n11, n12, n21, n22 = 0, 0, 0, 0
for sb, tf in zip(xgboostFeatures, mcnemarsfeatures):
    if sb == 1 and tf == 1:
        n11 += 1
    elif sb == 1 and tf == 0:
        n12 += 1
    elif sb == 0 and tf == 1:
        n21 += 1
    elif sb == 0 and tf == 0:
        n22 += 1

data = [[n11, n12], [n21, n22]]
print('xgboost and logreg features', mcnemar(data, exact=False, correction=True))

n11, n12, n21, n22 = 0, 0, 0, 0
for sb, tf in zip(xgboostTF, mcnemarstf):
    if sb == 1 and tf == 1:
        n11 += 1
    elif sb == 1 and tf == 0:
        n12 += 1
    elif sb == 0 and tf == 1:
        n21 += 1
    elif sb == 0 and tf == 0:
        n22 += 1

data = [[n11, n12], [n21, n22]]
print('xgboost and logreg tfidf', mcnemar(data, exact=False, correction=True))