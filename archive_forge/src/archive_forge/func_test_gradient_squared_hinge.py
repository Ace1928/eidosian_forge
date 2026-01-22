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
def test_gradient_squared_hinge():
    loss = sgd_fast.SquaredHinge(1.0)
    cases = [(1.0, 1.0, 0.0, 0.0), (-2.0, -1.0, 0.0, 0.0), (1.0, -1.0, 4.0, 4.0), (-1.0, 1.0, 4.0, -4.0), (0.5, 1.0, 0.25, -1.0), (0.5, -1.0, 2.25, 3.0)]
    _test_loss_common(loss, cases)