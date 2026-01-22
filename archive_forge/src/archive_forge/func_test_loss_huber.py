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
def test_loss_huber():
    loss = sgd_fast.Huber(0.1)
    cases = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.005, 0.1), (0.0, 0.1, 0.005, -0.1), (3.95, 4.0, 0.00125, -0.05), (5.0, 2.0, 0.295, 0.1), (-1.0, 5.0, 0.595, -0.1)]
    _test_loss_common(loss, cases)