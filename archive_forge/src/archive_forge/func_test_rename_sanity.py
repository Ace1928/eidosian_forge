import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test_rename_sanity():
    source_df = pandas.DataFrame(test_data['int_data'])[['col1', 'index', 'col3', 'col4']]
    mapping = {'col1': 'a', 'index': 'b', 'col3': 'c', 'col4': 'd'}
    modin_df = pd.DataFrame(source_df)
    df_equals(modin_df.rename(columns=mapping), source_df.rename(columns=mapping))
    renamed2 = source_df.rename(columns=str.lower)
    df_equals(modin_df.rename(columns=str.lower), renamed2)
    modin_df = pd.DataFrame(renamed2)
    df_equals(modin_df.rename(columns=str.upper), renamed2.rename(columns=str.upper))
    data = {'A': {'foo': 0, 'bar': 1}}
    df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    assert_index_equal(modin_df.rename(index={'foo': 'bar', 'bar': 'foo'}).index, df.rename(index={'foo': 'bar', 'bar': 'foo'}).index)
    assert_index_equal(modin_df.rename(index=str.upper).index, df.rename(index=str.upper).index)
    assert_index_equal(modin_df.rename(str.upper, axis=0).index, df.rename(str.upper, axis=0).index)
    assert_index_equal(modin_df.rename(str.upper, axis=1).columns, df.rename(str.upper, axis=1).columns)
    with pytest.raises(TypeError):
        modin_df.rename()
    source_df.rename(columns={'col3': 'foo', 'col4': 'bar'})
    modin_df = pd.DataFrame(source_df)
    assert_index_equal(modin_df.rename(columns={'col3': 'foo', 'col4': 'bar'}).index, source_df.rename(columns={'col3': 'foo', 'col4': 'bar'}).index)
    source_df.T.rename(index={'col3': 'foo', 'col4': 'bar'})
    assert_index_equal(source_df.T.rename(index={'col3': 'foo', 'col4': 'bar'}).index, modin_df.T.rename(index={'col3': 'foo', 'col4': 'bar'}).index)
    index = pandas.Index(['foo', 'bar'], name='name')
    renamer = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)
    renamed = renamer.rename(index={'foo': 'bar', 'bar': 'foo'})
    modin_renamed = modin_df.rename(index={'foo': 'bar', 'bar': 'foo'})
    assert_index_equal(renamed.index, modin_renamed.index)
    assert renamed.index.name == modin_renamed.index.name