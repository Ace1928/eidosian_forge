import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
Check that `predict` does return the expected output type.

    We need to check that `transform` will output a DataFrame and a NumPy array
    when we set `transform_output` to `pandas`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25499
    