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
@deprecated_parameter(deadline='invalid', fix='Double it yourself.', func_name='test_func', parameter_desc='double_count', match=lambda args, kwargs: 'double_count' in kwargs, rewrite=lambda args, kwargs: (args, {'new_count': kwargs['double_count'] * 2}))
def f_with_badly_deprecated_param(new_count):
    return new_count