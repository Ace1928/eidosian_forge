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
@given(sampled_from(reasonable_operators_for_scalars), sampled_from(types), sampled_from(types))
def test_operator_scalars(op, type1, type2):
    try:
        op(type1(1), type2(1))
    except TypeError:
        pass