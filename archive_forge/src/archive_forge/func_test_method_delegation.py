import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_method_delegation(self):
    ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
    result = ci.set_categories(list('cab'))
    tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cab')))
    ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
    result = ci.rename_categories(list('efg'))
    tm.assert_index_equal(result, CategoricalIndex(list('ffggef'), categories=list('efg')))
    result = ci.rename_categories(lambda x: x.upper())
    tm.assert_index_equal(result, CategoricalIndex(list('AABBCA'), categories=list('CAB')))
    ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
    result = ci.add_categories(['d'])
    tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cabd')))
    ci = CategoricalIndex(list('aabbca'), categories=list('cab'))
    result = ci.remove_categories(['c'])
    tm.assert_index_equal(result, CategoricalIndex(list('aabb') + [np.nan] + ['a'], categories=list('ab')))
    ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
    result = ci.as_unordered()
    tm.assert_index_equal(result, ci)
    ci = CategoricalIndex(list('aabbca'), categories=list('cabdef'))
    result = ci.as_ordered()
    tm.assert_index_equal(result, CategoricalIndex(list('aabbca'), categories=list('cabdef'), ordered=True))
    msg = 'cannot use inplace with CategoricalIndex'
    with pytest.raises(ValueError, match=msg):
        ci.set_categories(list('cab'), inplace=True)