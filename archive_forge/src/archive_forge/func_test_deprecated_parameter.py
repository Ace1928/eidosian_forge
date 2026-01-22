import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_deprecated_parameter():

    @deprecated_parameter(deadline='v1.2', fix='Double it yourself.', func_name='test_func', parameter_desc='double_count', match=lambda args, kwargs: 'double_count' in kwargs, rewrite=lambda args, kwargs: (args, {'new_count': kwargs['double_count'] * 2}))
    def f(new_count):
        return new_count
    with cirq.testing.assert_logs(count=0):
        assert f(1) == 1
        assert f(new_count=1) == 1
    with cirq.testing.assert_deprecated('_compat_test.py:', 'double_count parameter of test_func was used', 'will be removed in cirq v1.2', 'Double it yourself.', deadline='v1.2'):
        assert f(double_count=1) == 2
    with pytest.raises(ValueError, match='During testing using Cirq deprecated functionality is not allowed'):
        f(double_count=1)
    with pytest.raises(AssertionError, match='deadline should match vX.Y'):

        @deprecated_parameter(deadline='invalid', fix='Double it yourself.', func_name='test_func', parameter_desc='double_count', match=lambda args, kwargs: 'double_count' in kwargs, rewrite=lambda args, kwargs: (args, {'new_count': kwargs['double_count'] * 2}))
        def f_with_badly_deprecated_param(new_count):
            return new_count