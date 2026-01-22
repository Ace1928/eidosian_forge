import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.filterwarnings('ignore:Support for distance metrics:FutureWarning:sklearn')
def test_iris():
    for metric in ('euclidean', 'cosine'):
        clf = NearestCentroid(metric=metric).fit(iris.data, iris.target)
        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, 'Failed with score = ' + str(score)