import numpy as np
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn.preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys
from data import read_user

def dcg_at_k(y_score, k=5, method=0):
    y_score = np.asfarray(y_score)[:k]
    if y_score.size:
        if method == 0:
            return y_score[0] + np.sum(y_score[1:] / np.log2(np.arange(2, y_score.size + 1)))
        elif method == 1:
            return np.sum(y_score / np.log2(np.arange(2, y_score.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

def cal_NDCG_at_k(p, k=5, method=0):
    R_true = read_user('cf-test-1-users.dat')
    print(R_true.shape)
    dir_save = 'cdl' + str(p)
    U = np.mat(np.loadtxt(dir_save + '/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save + '/final-V.dat'))
    R = U * V.T
    num_u = R.shape[0]
    num_hit = 0
    user_id_sample = random.sample(list(range(num_u)), 300)

    valid_NDCGs = []
    for i in user_id_sample:
        idea_scores = np.ndarray.flatten(np.array(R_true[i, :]))
        idea_scores = idea_scores.astype(np.float32)
        idea_max = int(idea_scores.max())
        if idea_max < 1.0:
            continue
        item_scores = R[i, :].A1
        item_scores = item_scores.astype(np.float32)
        item_scores[item_scores < 0.0] = 0.0
        amin, amax = item_scores.min(), item_scores.max()
        y_score = (item_scores - amin) / (amax - amin)
        k = len(y_score)
        dcg_max = dcg_at_k(sorted(y_score, reverse=True), k, method)
        if not dcg_max:
            return 0.
        valid_NDCGs.append(dcg_at_k(y_score, k, method) / dcg_max)
    print("\nNDCG on (sampled) validation set: {}".format(np.mean(valid_NDCGs)))


cal_NDCG_at_k(4)


