import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def set_kmeans_params(self, **kwargs):
    k = kwargs.pop('n_clusters', False)
    if k:
        self.n_clusters = k
    self._sklearn_params = kwargs