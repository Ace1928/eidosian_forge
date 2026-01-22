import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
The provided value for alpha must only be
    used if alpha < _ALPHA_MIN and force_alpha is True.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/10772
    