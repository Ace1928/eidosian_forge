from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_skip_doctest():
    example = '>>> xxx\n>>>\n>>> # comment\n>>> xxx'
    res = skip_doctest(example)
    assert res == '>>> xxx  # doctest: +SKIP\n>>>\n>>> # comment\n>>> xxx  # doctest: +SKIP'
    assert skip_doctest(None) == ''
    example = '\n>>> 1 + 2  # doctest: +ELLIPSES\n3'
    expected = '\n>>> 1 + 2  # doctest: +ELLIPSES, +SKIP\n3'
    res = skip_doctest(example)
    assert res == expected