from collections import Iterable
import numpy as np
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from data import read_user
def cal_auc(p):
    R_true = read_user('cf-test-1-users.dat')
    print(R_true.shape)
    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    user_id_sample = random.sample(list(range(num_u)), 300)

    valid_AUCs = []
    for i in user_id_sample:
        idea_scores = np.ndarray.flatten(np.array(R_true[i, :]))
        idea_scores = idea_scores.astype(np.float32)
        idea_max = int(idea_scores.max())
        if idea_max < 1.0:
           continue
        item_scores = R[i,:].A1
        item_scores = item_scores.astype(np.float32)
        item_scores[item_scores < 0.0] = 0.0
        valid_AUCs.append(roc_auc_score(idea_scores, item_scores))
    print("\nAUC on (sampled) validation set: {}".format(np.mean(valid_AUCs)))

cal_auc(4)
