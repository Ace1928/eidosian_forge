import numpy as np
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def sparse_imputers():
    return [SimpleImputer()]