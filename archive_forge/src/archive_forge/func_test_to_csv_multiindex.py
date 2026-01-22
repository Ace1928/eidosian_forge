import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_multiindex(self, float_frame, datetime_frame):
    frame = float_frame
    old_index = frame.index
    arrays = np.arange(len(old_index) * 2, dtype=np.int64).reshape(2, -1)
    new_index = MultiIndex.from_arrays(arrays, names=['first', 'second'])
    frame.index = new_index
    with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:
        frame.to_csv(path, header=False)
        frame.to_csv(path, columns=['A', 'B'])
        frame.to_csv(path)
        df = self.read_csv(path, index_col=[0, 1], parse_dates=False)
        tm.assert_frame_equal(frame, df, check_names=False)
        assert frame.index.names == df.index.names
        float_frame.index = old_index
        tsframe = datetime_frame
        old_index = tsframe.index
        new_index = [old_index, np.arange(len(old_index), dtype=np.int64)]
        tsframe.index = MultiIndex.from_arrays(new_index)
        tsframe.to_csv(path, index_label=['time', 'foo'])
        with tm.assert_produces_warning(UserWarning, match='Could not infer format'):
            recons = self.read_csv(path, index_col=[0, 1], parse_dates=True)
        tm.assert_frame_equal(tsframe, recons, check_names=False)
        tsframe.to_csv(path)
        recons = self.read_csv(path, index_col=None)
        assert len(recons.columns) == len(tsframe.columns) + 2
        tsframe.to_csv(path, index=False)
        recons = self.read_csv(path, index_col=None)
        tm.assert_almost_equal(recons.values, datetime_frame.values)
        datetime_frame.index = old_index
    with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:

        def _make_frame(names=None):
            if names is True:
                names = ['first', 'second']
            return DataFrame(np.random.default_rng(2).integers(0, 10, size=(3, 3)), columns=MultiIndex.from_tuples([('bah', 'foo'), ('bah', 'bar'), ('ban', 'baz')], names=names), dtype='int64')
        df = DataFrame(np.ones((5, 3)), columns=MultiIndex.from_arrays([[f'i-{i}' for i in range(3)] for _ in range(4)], names=list('abcd')), index=MultiIndex.from_arrays([[f'i-{i}' for i in range(5)] for _ in range(2)], names=list('ab')))
        df.to_csv(path)
        result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1])
        tm.assert_frame_equal(df, result)
        df = DataFrame(np.ones((5, 3)), columns=MultiIndex.from_arrays([[f'i-{i}' for i in range(3)] for _ in range(4)], names=list('abcd')))
        df.to_csv(path)
        result = read_csv(path, header=[0, 1, 2, 3], index_col=0)
        tm.assert_frame_equal(df, result)
        df = DataFrame(np.ones((5, 3)), columns=MultiIndex.from_arrays([[f'i-{i}' for i in range(3)] for _ in range(4)], names=list('abcd')), index=MultiIndex.from_arrays([[f'i-{i}' for i in range(5)] for _ in range(3)], names=list('abc')))
        df.to_csv(path)
        result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1, 2])
        tm.assert_frame_equal(df, result)
        df = _make_frame()
        df.to_csv(path, index=False)
        result = read_csv(path, header=[0, 1])
        tm.assert_frame_equal(df, result)
        df = _make_frame(True)
        df.to_csv(path, index=False)
        result = read_csv(path, header=[0, 1])
        assert com.all_none(*result.columns.names)
        result.columns.names = df.columns.names
        tm.assert_frame_equal(df, result)
        df = _make_frame()
        df.to_csv(path)
        result = read_csv(path, header=[0, 1], index_col=[0])
        tm.assert_frame_equal(df, result)
        df = _make_frame(True)
        df.to_csv(path)
        result = read_csv(path, header=[0, 1], index_col=[0])
        tm.assert_frame_equal(df, result)
        df = _make_frame(True)
        df.to_csv(path)
        for i in [6, 7]:
            msg = f'len of {i}, but only 5 lines in file'
            with pytest.raises(ParserError, match=msg):
                read_csv(path, header=list(range(i)), index_col=0)
        msg = 'cannot specify cols with a MultiIndex'
        with pytest.raises(TypeError, match=msg):
            df.to_csv(path, columns=['foo', 'bar'])
    with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:
        tsframe[:0].to_csv(path)
        recons = self.read_csv(path)
        exp = tsframe[:0]
        exp.index = []
        tm.assert_index_equal(recons.columns, exp.columns)
        assert len(recons) == 0