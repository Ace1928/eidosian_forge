from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('constraint_declaration, expected_constraint_class', [(Interval(Real, 0, 1, closed='both'), Interval), (StrOptions({'option1', 'option2'}), StrOptions), (Options(Real, {0.42, 1.23}), Options), ('array-like', _ArrayLikes), ('sparse matrix', _SparseMatrices), ('random_state', _RandomStates), (None, _NoneConstraint), (callable, _Callables), (int, _InstancesOf), ('boolean', _Booleans), ('verbose', _VerboseHelper), (MissingValues(numeric_only=True), MissingValues), (HasMethods('fit'), HasMethods), ('cv_object', _CVObjects), ('nan', _NanConstraint)])
def test_make_constraint(constraint_declaration, expected_constraint_class):
    """Check that make_constraint dispatches to the appropriate constraint class"""
    constraint = make_constraint(constraint_declaration)
    assert constraint.__class__ is expected_constraint_class