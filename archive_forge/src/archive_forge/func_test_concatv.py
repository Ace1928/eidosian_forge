import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_concatv():
    assert list(concatv([], [], [])) == []
    assert list(take(5, concatv(['a', 'b'], range(1000000000)))) == ['a', 'b', 0, 1, 2]