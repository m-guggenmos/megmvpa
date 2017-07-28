import pickle
import os

import numpy as np
import scipy
import sklearn
from sklearn.discriminant_analysis import _cov
from sklearn.svm import SVC

np.random.seed(10)

preload_RDM = False

root = '/data/RSA/'

# Load data for each session
sessions = [
    dict(
        data=pickle.load(open(os.path.join(root, 'data01_sess1_bs10.pkl'), 'rb')),
        # data has shape n_trials x n_sensors x n_timepoints
        labels=pickle.load(open(os.path.join(root, 'labels01_sess1.pkl'), 'rb'))
        # labels has shape 1 x n_trials (i.e., one condition label [object category] per trial)
    ),
    dict(
        data=pickle.load(open(os.path.join(root, 'data01_sess2_bs10.pkl'), 'rb')),
        labels=pickle.load(open(os.path.join(root, 'labels01_sess2.pkl'), 'rb'))
    )
]

# Parameters
n_perm = 20
n_pseudo = 5
n_conditions = len(np.unique(sessions[0]['labels']))
n_sensors = sessions[0]['data'].shape[1]
n_time = sessions[0]['data'].shape[2]
n_sessions = len(sessions)


# Choose classifier
clf = SVC(kernel='linear')
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis()
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# from weird import WeiRD
# clf = WeiRD()

# Choose data partitioning scheme
from megmvpa.cv import ShuffleBinLeaveOneOut
CV = ShuffleBinLeaveOneOut


if preload_RDM:
    RDM = pickle.load(open(os.path.join(root, 'RDM.pkl'), 'rb'))
else:
    RDM = np.full((n_sessions, n_perm, n_conditions, n_conditions, n_time), np.nan)
    for s, session in enumerate(sessions):

        print('Session %g / %g' % (s + 1, n_sessions))

        X = session['data']
        y = session['labels']

        cv = CV(y, n_iter=n_perm, n_pseudo=n_pseudo)

        for f, (train_indices, test_indices) in enumerate(cv.split(X)):
            print('\tPermutation %g / %g' % (f + 1, n_perm))

            # average across trials to obtain pseudo-trials for training and test
            Xpseudo_train = np.full((len(train_indices), n_sensors, n_time), np.nan)
            Xpseudo_test = np.full((len(test_indices), n_sensors, n_time), np.nan)
            for i, ind in enumerate(train_indices):
                Xpseudo_train[i, :, :] = np.mean(X[ind, :, :], axis=0)
            for i, ind in enumerate(test_indices):
                Xpseudo_test[i, :, :] = np.mean(X[ind, :, :], axis=0)


            # Whitening using the Epoch method
            sigma_conditions = cv.labels_pseudo_train[0, :, n_pseudo-1:].flatten()
            sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
            for c in range(n_conditions):
                # compute sigma for each time point, then average across time
                sigma_[c] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, :, t], shrinkage='auto')
                                     for t in range(n_time)], axis=0)
            sigma = sigma_.mean(axis=0)  # average across conditions
            sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
            Xpseudo_train = (Xpseudo_train.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
            Xpseudo_test = (Xpseudo_test.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

            for t in range(n_time):
                for c1 in range(n_conditions-1):
                    for c2 in range(min(c1 + 1, n_conditions-1), n_conditions):
                            # fit the classifier using training data
                            data_train = Xpseudo_train[cv.ind_pseudo_train[c1, c2], :, t]
                            clf.fit(data_train, cv.labels_pseudo_train[c1, c2])

                            # compute predictions using test data
                            data_test = Xpseudo_test[cv.ind_pseudo_test[c1, c2], :, t]
                            predictions = clf.predict(data_test)

                            # compute dissimilarity and store in RDM
                            dissimilarity = np.mean(predictions == cv.labels_pseudo_test[c1, c2]) - 0.5
                            RDM[s, f, c1, c2, t] = np.mean(dissimilarity)
    # average across permutations
    RDM_av = np.nanmean(RDM, axis=1)
    pickle.dump(RDM_av, open(os.path.join(root, 'RDM.pkl'), 'wb'))

reliability_pearson = np.full(n_time, np.nan)
reliability_ssq = np.full(n_time, np.nan)
for t in range(n_time):
    d1 = RDM[0, :, :, t][np.isfinite(RDM[0, :, :, t])]
    d2 = RDM[1, :, :, t][np.isfinite(RDM[1, :, :, t])]
    reliability_pearson[t] = scipy.stats.pearsonr(d1, d2)[0]
    reliability_ssq[t] = 1 - np.sqrt(scipy.stats.stats._sum_of_squares(d1-d2)) / np.sqrt(np.sum(d1**2 + d2**2))



print('Finished!')