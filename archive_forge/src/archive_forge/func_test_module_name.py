import toolz
import toolz.curried
from toolz.curried import (take, first, second, sorted, merge_with, reduce,
from collections import defaultdict
from importlib import import_module
from operator import add
def test_module_name():
    assert toolz.curried.__name__ == 'toolz.curried'