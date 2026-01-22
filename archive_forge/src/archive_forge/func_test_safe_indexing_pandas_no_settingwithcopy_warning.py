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
def test_safe_indexing_pandas_no_settingwithcopy_warning():
    pd = pytest.importorskip('pandas')
    pd_version = parse_version(pd.__version__)
    pd_base_version = parse_version(pd_version.base_version)
    if pd_base_version >= parse_version('3'):
        raise SkipTest('SettingWithCopyWarning has been removed in pandas 3.0.0.dev')
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
    subset = _safe_indexing(X, [0, 1], axis=0)
    if hasattr(pd.errors, 'SettingWithCopyWarning'):
        SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
    else:
        SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', SettingWithCopyWarning)
        subset.iloc[0, 0] = 10
    assert X.iloc[0, 0] == 1