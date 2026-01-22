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
def test_none_iter():
    st = SelfTrainingClassifier(KNeighborsClassifier(), threshold=0.55, max_iter=None)
    st.fit(X_train, y_train_missing_labels)
    assert st.n_iter_ < 10
    assert st.termination_condition_ == 'all_labeled'