import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_proxy_with_pandas_constructor(self):
    """Verify that users still can use pandas' constructor using `type(cat)(...)` notation."""
    lazy_proxy, _, _ = self._get_lazy_proxy()
    assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
    new_cat_values = pandas.Index([3, 4, 5])
    new_category_dtype = type(lazy_proxy)(categories=new_cat_values, ordered=True)
    assert not lazy_proxy._is_materialized
    assert new_category_dtype._is_materialized
    assert new_category_dtype.categories.equals(new_cat_values)
    assert new_category_dtype.ordered