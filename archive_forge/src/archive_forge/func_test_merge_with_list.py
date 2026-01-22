import toolz
import toolz.curried
from toolz.curried import (take, first, second, sorted, merge_with, reduce,
from collections import defaultdict
from importlib import import_module
from operator import add
def test_merge_with_list():
    assert merge_with(sum, [{'a': 1}, {'a': 2}]) == {'a': 3}