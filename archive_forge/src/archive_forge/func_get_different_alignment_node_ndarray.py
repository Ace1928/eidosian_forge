import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def get_different_alignment_node_ndarray(node_ndarray):
    new_dtype_dict = {name: dtype for name, (dtype, _) in node_ndarray.dtype.fields.items()}
    offsets = [offset for dtype, offset in node_ndarray.dtype.fields.values()]
    shifted_offsets = [8 + offset for offset in offsets]
    new_dtype = np.dtype({'names': list(new_dtype_dict.keys()), 'formats': list(new_dtype_dict.values()), 'offsets': shifted_offsets})
    return node_ndarray.astype(new_dtype, casting='same_kind')