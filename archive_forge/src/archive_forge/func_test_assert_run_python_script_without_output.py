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
@pytest.mark.xfail(_IS_WASM, reason='cannot start subprocess')
def test_assert_run_python_script_without_output():
    code = 'x = 1'
    assert_run_python_script_without_output(code)
    code = "print('something to stdout')"
    with pytest.raises(AssertionError, match='Expected no output'):
        assert_run_python_script_without_output(code)
    code = "print('something to stdout')"
    with pytest.raises(AssertionError, match='output was not supposed to match.+got.+something to stdout'):
        assert_run_python_script_without_output(code, pattern='to.+stdout')
    code = '\n'.join(['import sys', "print('something to stderr', file=sys.stderr)"])
    with pytest.raises(AssertionError, match='output was not supposed to match.+got.+something to stderr'):
        assert_run_python_script_without_output(code, pattern='to.+stderr')