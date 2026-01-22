import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_gen_even_slices():
    some_range = range(10)
    joined_range = list(chain(*[some_range[slice] for slice in gen_even_slices(10, 3)]))
    assert_array_equal(some_range, joined_range)