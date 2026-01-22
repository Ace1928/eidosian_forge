import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def eval_rolling(md_window, pd_window):
    eval_general(md_window, pd_window, lambda window: window.count())
    eval_general(md_window, pd_window, lambda window: window.sum())
    eval_general(md_window, pd_window, lambda window: window.mean())
    eval_general(md_window, pd_window, lambda window: window.median())
    eval_general(md_window, pd_window, lambda window: window.var())
    eval_general(md_window, pd_window, lambda window: window.std())
    eval_general(md_window, pd_window, lambda window: window.min())
    eval_general(md_window, pd_window, lambda window: window.max())
    expected_exception = None
    if pd_window.on == 'col4':
        expected_exception = ValueError('Length mismatch: Expected axis has 450 elements, new values have 600 elements')
    eval_general(md_window, pd_window, lambda window: window.corr(), expected_exception=expected_exception)
    eval_general(md_window, pd_window, lambda window: window.cov(), expected_exception=expected_exception)
    eval_general(md_window, pd_window, lambda window: window.skew())
    eval_general(md_window, pd_window, lambda window: window.kurt())
    eval_general(md_window, pd_window, lambda window: window.apply(lambda df: (df + 10).sum()))
    eval_general(md_window, pd_window, lambda window: window.agg('sum'))
    eval_general(md_window, pd_window, lambda window: window.quantile(0.2))
    eval_general(md_window, pd_window, lambda window: window.rank())
    expected_exception = None
    if pd_window.on == 'col4':
        expected_exception = TypeError('Addition/subtraction of integers and integer-arrays with DatetimeArray is no longer supported.' + '  Instead of adding/subtracting `n`, use `n * obj.freq`')
    if not md_window._as_index:
        by_cols = list(md_window._groupby_obj._internal_by)
        eval_general(md_window, pd_window, lambda window: window.sem().drop(columns=by_cols, errors='ignore'), expected_exception=expected_exception)
    else:
        eval_general(md_window, pd_window, lambda window: window.sem(), expected_exception=expected_exception)