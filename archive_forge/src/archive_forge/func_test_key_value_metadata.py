from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_key_value_metadata():
    m = pa.KeyValueMetadata({'a': 'A', 'b': 'B'})
    assert len(m) == 2
    assert m['a'] == b'A'
    assert m[b'a'] == b'A'
    assert m['b'] == b'B'
    assert 'a' in m
    assert b'a' in m
    assert 'c' not in m
    m1 = pa.KeyValueMetadata({'a': 'A', 'b': 'B'})
    m2 = pa.KeyValueMetadata(a='A', b='B')
    m3 = pa.KeyValueMetadata([('a', 'A'), ('b', 'B')])
    assert m1 != 2
    assert m1 == m2
    assert m2 == m3
    assert m1 == {'a': 'A', 'b': 'B'}
    assert m1 != {'a': 'A', 'b': 'C'}
    with pytest.raises(TypeError):
        pa.KeyValueMetadata({'a': 1})
    with pytest.raises(TypeError):
        pa.KeyValueMetadata({1: 'a'})
    with pytest.raises(TypeError):
        pa.KeyValueMetadata(a=1)
    expected = [(b'a', b'A'), (b'b', b'B')]
    result = [(k, v) for k, v in m3.items()]
    assert result == expected
    assert list(m3.items()) == expected
    assert list(m3.keys()) == [b'a', b'b']
    assert list(m3.values()) == [b'A', b'B']
    assert len(m3) == 2
    md = pa.KeyValueMetadata([('a', 'alpha'), ('b', 'beta'), ('a', 'Alpha'), ('a', 'ALPHA')])
    expected = [(b'a', b'alpha'), (b'b', b'beta'), (b'a', b'Alpha'), (b'a', b'ALPHA')]
    assert len(md) == 4
    assert isinstance(md.keys(), Iterator)
    assert isinstance(md.values(), Iterator)
    assert isinstance(md.items(), Iterator)
    assert list(md.items()) == expected
    assert list(md.keys()) == [k for k, _ in expected]
    assert list(md.values()) == [v for _, v in expected]
    assert md['a'] == b'alpha'
    assert md['b'] == b'beta'
    assert md.get_all('a') == [b'alpha', b'Alpha', b'ALPHA']
    assert md.get_all('b') == [b'beta']
    assert md.get_all('unknown') == []
    with pytest.raises(KeyError):
        md = pa.KeyValueMetadata([('a', 'alpha'), ('b', 'beta'), ('a', 'Alpha'), ('a', 'ALPHA')], b='BETA')