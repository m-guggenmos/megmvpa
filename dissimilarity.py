import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_cov, _class_means, \
    linalg


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