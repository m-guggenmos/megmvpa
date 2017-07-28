import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.classification import _weighted_sum


class WeiRD(BaseEstimator, ClassifierMixin):
    """WeiRD - weighted robust distance classifier

    WeiRD stands for "Weighted Robust Distance" and is a fast and simple classification algorithm
    that assigns class labels based on the distance to class prototypes_. The distance is the
    Manhattan or Euclidian distance between a current sample and a prototype in a space, in which
    each feature dimension is scaled by the two-sample t-value of the respective feature in the
    training data. Class prototypes_ correspond to the arithmetic prototypes_ of each feature in the
    training data. The current implementation works for two-class problems only.
    __________________________________________________________________________
    Matthias Guggenmos, Katharina Schmack and Philipp Sterzer, "WeiRD - a fast and performant
    multivariate pattern classifier," 2016 International Workshop on Pattern Recognition in
    Neuroimaging (PRNI), Trento, Italy, 2016, pp. 1-4. doi: 10.1109/PRNI.2016.7552349

    Example:
        import numpy as np
        from weird import WeiRD

        # parameters
        n_samples_per_class = 100
        n_features = 20

        # create data
        X1 = np.random.rand(n_features) + np.random.rand(n_samples_per_class, n_features)
        X2 = np.random.rand(n_features) + np.random.rand(n_samples_per_class, n_features)
        X_fit = np.vstack((X1, X2))
        X_predict = X_fit + np.random.rand(2*n_samples_per_class, n_features)
        y = np.hstack((np.zeros(n_samples_per_class), np.ones(n_samples_per_class)))

        # perform classification
        weird = WeiRD()
        weird.fit(X_fit, y)
        predictions = weird.predict(X_predict)
        print('Classification accuracy = %.1f%%' % (100*np.mean((predictions == y))))

    """

    def __init__(self, centroid_weighting=True, stats_weighting=True, exponential=False,
                 distance_type='manhattan', verbose=0):
        """
        Args:
            centroid_weighting (boolean): If true, the gradual distance of a new sample to the
                centroids is considered, else it is only considered to which of the two centroids
                the sample is closer, i.e. a binary measure. Defaults to True.
            stats_weighting (boolean): Switch on 'statistical' weighting, i.e. scaling the feature
                space with independent t-test values from the training data. Defaults to True.
            exponential (boolean): Scale feature importances exponentially. Defaults to False.
            distance_type (str): if 'manhattan', compute distances to prototypes_ using the
                Manhattan distance  (L1 norm); if 'euclidean', compute distances to prototypes_ using
                the Euclidean distance. Defaults to 'manhattan'.
            verbose (int): Legacy parameter without any function at present. Defaults to 0.
        """
        self.centroid_weighting = centroid_weighting
        self.stats_weighting = stats_weighting
        self.exponential = exponential
        self.distance_type = distance_type
        self.verbose = verbose

        self.classes_ = None
        self.feature_importances_ = None
        self.prototypes_ = None

    def fit(self, X, y):
        """ Train the model.

        Args:
            X (np.ndarray, List): Data in the form of rows x columns = samples x features.
            y (np.ndarray, List): Class labels, one value per row of X.

        Returns:
            the class instance
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        x1 = X[np.array(y) == self.classes_[0], :]
        x2 = X[np.array(y) == self.classes_[1], :]

        self.prototypes_ = np.vstack((x1.mean(axis=0), x2.mean(axis=0)))

        if self.stats_weighting:
            statistic = _ttest_ind(x1, x2, self.prototypes_)
            statistic[np.isnan(statistic)] = 0
            self.feature_importances_ = np.atleast_1d(abs(statistic[:, np.newaxis]).squeeze())
            if self.exponential:
                self.feature_importances_ = np.exp(self.feature_importances_)
        else:
            self.feature_importances_ = np.ones(X.shape[1])

        return self

    def predict(self, X):
        """ Predict new samples based on the trained model.

        Args:
            X (np.ndarray, List): Data in the form of rows x columns = samples x features.

        Returns:
            np.ndarray: Predicted class labels.
        """

        dec = self.decision_function(X)
        return self.classes_[(dec > 0).astype(int)]

    def decision_function(self, X):
        """ Compute the (weighted) sum of votes.

        Args:
            X (np.ndarray, List): Data in the form of rows x columns = samples x features.

        Returns:
            np.ndarray: The (weighted) sum of votes for each sample in the form 1 x samples.
        """

        X = np.array(X)

        if self.distance_type == 'manhattan':
            if self.centroid_weighting:
                self.votes_ = abs(X - self.prototypes_[0, :]) - abs(X - self.prototypes_[1, :])
            else:
                self.votes_ = (abs(X - self.prototypes_[0, :]) > abs(X - self.prototypes_[1, :])) - 0.5
            dec = _weighted_sum(self.votes_, self.feature_importances_) / self.votes_.shape[1]
        elif self.distance_type == 'euclidean':
            dec = np.sum((self.feature_importances_ * (X - self.prototypes_[0, :])) ** 2, axis=1) - \
                  np.sum((self.feature_importances_ * (X - self.prototypes_[1, :])) ** 2, axis=1)

        return dec


def _ttest_ind(x1, x2, means):
    """ Efficient implementation of a two-sample t-test

        Args:
            x1 (np.ndarray): Data of class 1 in the form rows x columns = samples x features.
            x2 (np.ndarray): Data of class 2 in the form rows x columns = samples x features.
            means (np.ndarray): Mean values for each feature in the form 1 x features
                (corresponds to prototypes)

        Returns:
            np.ndarray: two-sample t-test values for each feature
    """

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    gsd = np.sqrt(((n1 - 1) * np.nanvar(x1) + (n2 - 1) * np.nanvar(x2)) / (n1 + n2 - 2))
    t = (means[0] - means[1]) / (gsd * np.sqrt(1 / n1 + 1 / n2))

    return t

