import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
def test_assert_raise_message():

    def _raise_ValueError(message):
        raise ValueError(message)

    def _no_raise():
        pass
    assert_raise_message(ValueError, 'test', _raise_ValueError, 'test')
    assert_raises(AssertionError, assert_raise_message, ValueError, 'something else', _raise_ValueError, 'test')
    assert_raises(ValueError, assert_raise_message, TypeError, 'something else', _raise_ValueError, 'test')
    assert_raises(AssertionError, assert_raise_message, ValueError, 'test', _no_raise)
    assert_raises(AssertionError, assert_raise_message, (ValueError, AttributeError), 'test', _no_raise)