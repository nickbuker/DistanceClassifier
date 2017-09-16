import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean


class DistanceClassifier(object):
    def __init__(self, l=0):
        """
        l for compatibility with sklearn tools
        """
        self.l = l

    def fit(self, X_train, y_train):
        """
        Call _find_centroids to locate the centroids of each class

        Input:
        X_train (pandas Dataframe or numpy array) data,
        y_train (pandas Series or numpy array) labels

        Returns:
        None
        """
        self._cent_map, self._centroids = self._find_centroids(X_train, y_train)

    def _find_centroids(self, X_train, y_train):
        """
        Locate the centroids of each class

        Input:
        X_train (pandas Dataframe or numpy array) data,
        y_train (pandas Series or numpy array) labels

        Returns:
        cent_map (dictionary with column names:labels),
        centroids (list of pandas Series) coordinates of centroids
        """
        centroids = []
        cent_map = {}
        for i, c in enumerate(y_train.unique()):
            centroids.append(np.mean(X_train[y_train == c]))
            cent_map['c{}'.format(i)] = c
        return cent_map, centroids

    def predict(self, X_test):
        """
        Calls _find_distances and _assign_points to classify test data

        Input:
        X_test (pandas DataFrame or numpy array) data

        Returns:
        pandas Series of assignments
        """
        distances = self._find_distances(X_test)
        return self._assign_points(distances)

    def _find_distances(self, X_test):
        """
        Calculate euclidean distance between the centroids and data points

        Input:
        X_test (pandas Dataframe or numpy array) data

        Returns:
        pandas DataFrame with distances to each centroid
        """
        # wrap euclidean distance to work with apply
        def euc(series):
            return euclidean(series, c)

        distances = pd.DataFrame()
        for i, c in enumerate(self._centroids):
            distances['c{}'.format(i)] = X_test.apply(euc, axis=1)
        return distances

    def _assign_points(self, distances):
        """
        Use distances and _cent_map to assign labels to points

        Input:
        distances to each centroid (pandas DataFrame)

        Returns:
        pandas Series of assignments
        """
        cent_cols = distances.idxmin(axis=1)
        return pd.Series([self._cent_map[p] for p in cent_cols])

    def get_params(self, deep=False):
        """
        This method is for sklearn compatibility
        """
        return {'l': self.l}
