from math import ceil
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
def test_strings_dtype():
    clf = SelfTrainingClassifier(KNeighborsClassifier())
    X, y = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)
    labels_multiclass = ['one', 'two', 'three']
    y_strings = np.take(labels_multiclass, y)
    with pytest.raises(ValueError, match='dtype'):
        clf.fit(X, y_strings)