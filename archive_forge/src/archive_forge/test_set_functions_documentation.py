import pytest
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
from numpy import array_api as xp

    Inverse indices share shape of input array

    See https://github.com/numpy/numpy/issues/20638
    