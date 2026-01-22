import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@given(sampled_from(objecty_things), sampled_from(reasonable_operators_for_scalars), sampled_from(types))
def test_operator_object_left(o, op, type_):
    try:
        with recursionlimit(200):
            op(o, type_(1))
    except TypeError:
        pass