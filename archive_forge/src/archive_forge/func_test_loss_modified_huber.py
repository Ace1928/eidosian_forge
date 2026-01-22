import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
def test_loss_modified_huber():
    loss = sgd_fast.ModifiedHuber()
    cases = [(1.0, 1.0, 0.0, 0.0), (-1.0, -1.0, 0.0, 0.0), (2.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, -2.0), (-1.0, 1.0, 4.0, -4.0), (0.5, -1.0, 2.25, 3.0), (-2.0, 1.0, 8, -4.0), (-3.0, 1.0, 12, -4.0)]
    _test_loss_common(loss, cases)