import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_cov, _class_means, \
    linalg
from sklearn.base import BaseEstimator
from scipy.spatial.distance import euclidean, correlation


class Euclidean2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return euclidean(np.mean(X[y == self.classes_[0]], axis=0),
                         np.mean(X[y == self.classes_[1]], axis=0))**2


class CvEuclidean2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train @ dist_test


class Pearson(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, return_1d=False):

        self.random_state = random_state
        self.verbose = verbose
        self.return_1d = return_1d

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        d = correlation(np.mean(X[y == self.classes_[0]], axis=0),
                        np.mean(X[y == self.classes_[1]], axis=0))

        if self.return_1d:
            d = np.atleast_1d(d)

        return d


class CvPearson(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, regularize_var=True, regularize_denom=True,
                 reg_factor_var=0.1, reg_factor_denom=0.25, bounded=True, reg_bounding=1,
                 return_1d=False):

        self.random_state = random_state
        self.verbose = verbose
        self.regularize_var = regularize_var
        self.regularize_denom = regularize_denom
        self.reg_factor_var = reg_factor_var
        self.reg_factor_denom = reg_factor_denom
        self.bounded = bounded
        self.reg_bounding = reg_bounding
        self.return_1d = return_1d

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)
        self.var_A1 = np.var(self.A1)
        self.var_B1 = np.var(self.B1)
        self.denom_noncv = np.sqrt(self.var_A1 * self.var_B1)

        return self

    def predict(self, X, y=None):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        cov_a1b2 = np.cov(self.A1, B2)[0, 1]
        cov_b1a2 = np.cov(self.B1, A2)[0, 1]
        cov_ab = (cov_a1b2 + cov_b1a2) / 2

        var_A12 = np.cov(self.A1, A2)[0, 1]
        var_B12 = np.cov(self.B1, B2)[0, 1]

        if self.regularize_var:
            denom = np.sqrt(max(self.reg_factor_var * self.var_A1, var_A12) * max(self.reg_factor_var * self.var_B1, var_B12))
        else:
            denom = np.sqrt(var_A12 * var_B12)
        if self.regularize_denom:
            denom = max(self.reg_factor_denom * self.denom_noncv, denom)

        r = cov_ab / denom

        if self.bounded:
            r = min(max(-self.reg_bounding, r), self.reg_bounding)

        d = 1 - r
        if self.return_1d:
            d = np.atleast_1d(d)
        return d



class LDA(LinearDiscriminantAnalysis):
    """Wrapper to sklearn.discriminant_analysis.LinearDiscriminantAnalysis which allows passing
    a custom covariance matrix (sigma)
    """

    def __init__(self, solver='lsqr', shrinkage=None, priors=None, n_components=None,
                 tol=1e-4, sigma=None):

        super().__init__(solver=solver, shrinkage=shrinkage, priors=priors,
                         n_components=n_components, tol=tol)

        self.sigma = sigma

    def _solve_lsqr(self, X, y, shrinkage):

        self.means_ = _class_means(X, y)
        if self.sigma is not None:
            self.covariance_ = self.sigma
        else:
            self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T))
                           + np.log(self.priors_))