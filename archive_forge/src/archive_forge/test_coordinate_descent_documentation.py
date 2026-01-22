import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
Check that for :class:`MultiTaskElasticNetCV` and
    class:`MultiTaskLassoCV` if `sample_weight` is passed and the
    CV splitter does not support `sample_weight` an error is raised.
    On the other hand if the splitter does support `sample_weight`
    while `sample_weight` is passed there is no error and process
    completes smoothly as before.
    