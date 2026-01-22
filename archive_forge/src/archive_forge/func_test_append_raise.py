import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
@td.skip_array_manager_not_yet_implemented
def test_append_raise(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        df['invalid'] = [['a']] * len(df)
        assert df.dtypes['invalid'] == np.object_
        msg = re.escape('Cannot serialize the column [invalid]\nbecause its data contents are not [string] but [mixed] object dtype')
        with pytest.raises(TypeError, match=msg):
            store.append('df', df)
        df['invalid2'] = [['a']] * len(df)
        df['invalid3'] = [['a']] * len(df)
        with pytest.raises(TypeError, match=msg):
            store.append('df', df)
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        s = Series(datetime.datetime(2001, 1, 2), index=df.index)
        s = s.astype(object)
        s[0:5] = np.nan
        df['invalid'] = s
        assert df.dtypes['invalid'] == np.object_
        msg = 'too many timezones in this block, create separate data columns'
        with pytest.raises(TypeError, match=msg):
            store.append('df', df)
        msg = 'value must be None, Series, or DataFrame'
        with pytest.raises(TypeError, match=msg):
            store.append('df', np.arange(10))
        msg = re.escape("cannot properly create the storer for: [group->df,value-><class 'pandas.core.series.Series'>]")
        with pytest.raises(TypeError, match=msg):
            store.append('df', Series(np.arange(10)))
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        store.append('df', df)
        df['foo'] = 'foo'
        msg = re.escape("invalid combination of [non_index_axes] on appending data [(1, ['A', 'B', 'C', 'D', 'foo'])] vs current table [(1, ['A', 'B', 'C', 'D'])]")
        with pytest.raises(ValueError, match=msg):
            store.append('df', df)
        _maybe_remove(store, 'df')
        df['foo'] = Timestamp('20130101')
        store.append('df', df)
        df['foo'] = 'bar'
        msg = re.escape('invalid combination of [values_axes] on appending data [name->values_block_1,cname->values_block_1,dtype->bytes24,kind->string,shape->(1, 30)] vs current table [name->values_block_1,cname->values_block_1,dtype->datetime64[s],kind->datetime64[s],shape->None]')
        with pytest.raises(ValueError, match=msg):
            store.append('df', df)