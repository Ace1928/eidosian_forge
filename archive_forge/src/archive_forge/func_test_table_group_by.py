from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_table_group_by():

    def sorted_by_keys(d):
        if 'keys2' in d:
            keys = tuple(zip(d['keys'], d['keys2']))
        else:
            keys = d['keys']
        sorted_keys = sorted(keys)
        sorted_d = {'keys': sorted(d['keys'])}
        for entry in d:
            if entry == 'keys':
                continue
            values = dict(zip(keys, d[entry]))
            for k in sorted_keys:
                sorted_d.setdefault(entry, []).append(values[k])
        return sorted_d
    table = pa.table([pa.array(['a', 'a', 'b', 'b', 'c']), pa.array(['X', 'X', 'Y', 'Z', 'Z']), pa.array([1, 2, 3, 4, 5]), pa.array([10, 20, 30, 40, 50])], names=['keys', 'keys2', 'values', 'bigvalues'])
    r = table.group_by('keys').aggregate([('values', 'hash_sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'c'], 'values_sum': [3, 7, 5]}
    r = table.group_by('keys').aggregate([('values', 'hash_sum'), ('values', 'hash_count')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'c'], 'values_sum': [3, 7, 5], 'values_count': [2, 2, 1]}
    r = table.group_by('keys').aggregate([('values', 'sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'c'], 'values_sum': [3, 7, 5]}
    r = table.group_by('keys').aggregate([('values', 'max'), ('bigvalues', 'sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'c'], 'values_max': [2, 4, 5], 'bigvalues_sum': [30, 70, 50]}
    r = table.group_by('keys').aggregate([('bigvalues', 'max'), ('values', 'sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'c'], 'values_sum': [3, 7, 5], 'bigvalues_max': [20, 40, 50]}
    r = table.group_by(['keys', 'keys2']).aggregate([('values', 'sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'b', 'c'], 'keys2': ['X', 'Y', 'Z', 'Z'], 'values_sum': [3, 3, 4, 5]}
    r = table.group_by('keys').aggregate([('values', 'max'), ('bigvalues', 'sum'), ('bigvalues', 'max'), ([], 'count_all'), ('values', 'sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b', 'c'], 'values_max': [2, 4, 5], 'bigvalues_sum': [30, 70, 50], 'bigvalues_max': [20, 40, 50], 'count_all': [2, 2, 1], 'values_sum': [3, 7, 5]}
    table_with_nulls = pa.table([pa.array(['a', 'a', 'a']), pa.array([1, None, None])], names=['keys', 'values'])
    r = table_with_nulls.group_by(['keys']).aggregate([('values', 'count', pc.CountOptions(mode='all'))])
    assert r.to_pydict() == {'keys': ['a'], 'values_count': [3]}
    r = table_with_nulls.group_by(['keys']).aggregate([('values', 'count', pc.CountOptions(mode='only_null'))])
    assert r.to_pydict() == {'keys': ['a'], 'values_count': [2]}
    r = table_with_nulls.group_by(['keys']).aggregate([('values', 'count', pc.CountOptions(mode='only_valid'))])
    assert r.to_pydict() == {'keys': ['a'], 'values_count': [1]}
    r = table_with_nulls.group_by(['keys']).aggregate([([], 'count_all'), ('values', 'count', pc.CountOptions(mode='only_valid'))])
    assert r.to_pydict() == {'keys': ['a'], 'count_all': [3], 'values_count': [1]}
    r = table_with_nulls.group_by(['keys']).aggregate([([], 'count_all')])
    assert r.to_pydict() == {'keys': ['a'], 'count_all': [3]}
    table = pa.table({'keys': ['a', 'b', 'a', 'b', 'a', 'b'], 'values': range(6)})
    table_with_chunks = pa.Table.from_batches(table.to_batches(max_chunksize=3))
    r = table_with_chunks.group_by('keys').aggregate([('values', 'sum')])
    assert sorted_by_keys(r.to_pydict()) == {'keys': ['a', 'b'], 'values_sum': [6, 9]}