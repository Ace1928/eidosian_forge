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
def test_get_chunk_n_rows_warns():
    """Check that warning is raised when working_memory is too low."""
    row_bytes = 1024 * 1024 + 1
    max_n_rows = None
    working_memory = 1
    expected = 1
    warn_msg = 'Could not adhere to working_memory config. Currently 1MiB, 2MiB required.'
    with pytest.warns(UserWarning, match=warn_msg):
        actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows, working_memory=working_memory)
    assert actual == expected
    assert type(actual) is type(expected)
    with config_context(working_memory=working_memory):
        with pytest.warns(UserWarning, match=warn_msg):
            actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows)
        assert actual == expected
        assert type(actual) is type(expected)