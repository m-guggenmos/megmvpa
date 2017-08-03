import numpy as np

class ShuffleBin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                       np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                          np.nan, dtype=np.int)

        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)
            for c2 in range(self.n_classes):
                range_c2 = range(c2*self.n_pseudo, (c2+1)*self.n_pseudo)
                self.ind_pseudo_test[c1, c2, :2 * self.n_pseudo] = np.concatenate((range_c1, range_c2))
                self.labels_pseudo_test[c1, c2, :2 * self.n_pseudo] = \
                    np.concatenate((self.classes[c1] * np.ones(self.n_pseudo),
                                    self.classes[c2] * np.ones(self.n_pseudo)))
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=np.object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self):
        return self.n_iter


class ShuffleBinLeaveOneOut:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_train = np.full((self.n_classes, self.n_classes, 2*(self.n_pseudo-1)),
                                        np.nan, dtype=np.int)
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2), np.nan, dtype=np.int)
        self.labels_pseudo_train = np.full((self.n_classes, self.n_classes, 2*(self.n_pseudo-1)),
                                           np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2), np.nan, dtype=np.int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*(self.n_pseudo-1), (c1+1)*(self.n_pseudo-1))
            for c2 in range(self.n_classes):
                range_c2 = range(c2*(self.n_pseudo-1), (c2+1)*(self.n_pseudo-1))
                self.ind_pseudo_train[c1, c2, :2*(self.n_pseudo - 1)] = \
                    np.concatenate((range_c1, range_c2))
                self.ind_pseudo_test[c1, c2] = [c1, c2]

                self.labels_pseudo_train[c1, c2, :2*(self.n_pseudo - 1)] = \
                    np.concatenate((self.classes[c1] * np.ones(self.n_pseudo - 1),
                                    self.classes[c2] * np.ones(self.n_pseudo - 1)))
                self.labels_pseudo_test[c1, c2] = self.classes[[c1, c2]].astype(self.labels_pseudo_train.dtype)

    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_train = np.full(self.n_classes*(self.n_pseudo-1), np.nan, dtype=np.object)
        _ind_test = np.full(self.n_classes, np.nan, dtype=np.object)
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*(self.n_pseudo-1), (c1+1)*(self.n_pseudo-1))):
                    _ind_train[j] = ind[i]
                _ind_test[c1] = ind[-1]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def get_n_splits(self):
        return self.n_iter

    def __len__(self):
        return self.n_iter

