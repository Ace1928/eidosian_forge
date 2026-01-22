from collections import OrderedDict
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.skipif(using_pyarrow_string_dtype(), reason='columns inferring logic broken')
def test_constructor_list_of_series(self):
    data = [OrderedDict([['a', 1.5], ['b', 3.0], ['c', 4.0]]), OrderedDict([['a', 1.5], ['b', 3.0], ['c', 6.0]])]
    sdict = OrderedDict(zip(['x', 'y'], data))
    idx = Index(['a', 'b', 'c'])
    data2 = [Series([1.5, 3, 4], idx, dtype='O', name='x'), Series([1.5, 3, 6], idx, name='y')]
    result = DataFrame(data2)
    expected = DataFrame.from_dict(sdict, orient='index')
    tm.assert_frame_equal(result, expected)
    data2 = [Series([1.5, 3, 4], idx, dtype='O', name='x'), Series([1.5, 3, 6], idx)]
    result = DataFrame(data2)
    sdict = OrderedDict(zip(['x', 'Unnamed 0'], data))
    expected = DataFrame.from_dict(sdict, orient='index')
    tm.assert_frame_equal(result, expected)
    data = [OrderedDict([['a', 1.5], ['b', 3], ['c', 4], ['d', 6]]), OrderedDict([['a', 1.5], ['b', 3], ['d', 6]]), OrderedDict([['a', 1.5], ['d', 6]]), OrderedDict(), OrderedDict([['a', 1.5], ['b', 3], ['c', 4]]), OrderedDict([['b', 3], ['c', 4], ['d', 6]])]
    data = [Series(d) for d in data]
    result = DataFrame(data)
    sdict = OrderedDict(zip(range(len(data)), data))
    expected = DataFrame.from_dict(sdict, orient='index')
    tm.assert_frame_equal(result, expected.reindex(result.index))
    result2 = DataFrame(data, index=np.arange(6, dtype=np.int64))
    tm.assert_frame_equal(result, result2)
    result = DataFrame([Series(dtype=object)])
    expected = DataFrame(index=[0])
    tm.assert_frame_equal(result, expected)
    data = [OrderedDict([['a', 1.5], ['b', 3.0], ['c', 4.0]]), OrderedDict([['a', 1.5], ['b', 3.0], ['c', 6.0]])]
    sdict = OrderedDict(zip(range(len(data)), data))
    idx = Index(['a', 'b', 'c'])
    data2 = [Series([1.5, 3, 4], idx, dtype='O'), Series([1.5, 3, 6], idx)]
    result = DataFrame(data2)
    expected = DataFrame.from_dict(sdict, orient='index')
    tm.assert_frame_equal(result, expected)