import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
def custom_callable(x, y, missing_values=np.nan, squared=False):
    x = np.ma.array(x, mask=np.isnan(x))
    y = np.ma.array(y, mask=np.isnan(y))
    dist = np.nansum(np.abs(x - y))
    return dist