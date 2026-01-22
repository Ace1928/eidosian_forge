from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_update_in(self):
    D, kw = (self.D, self.kw)
    assert update_in(D({'a': 0}), ['a'], inc, **kw) == D({'a': 1})
    assert update_in(D({'a': 0, 'b': 1}), ['b'], str, **kw) == D({'a': 0, 'b': '1'})
    assert update_in(D({'t': 1, 'v': D({'a': 0})}), ['v', 'a'], inc, **kw) == D({'t': 1, 'v': D({'a': 1})})
    assert update_in(D({}), ['z'], str, None, **kw) == D({'z': 'None'})
    assert update_in(D({}), ['z'], inc, 0, **kw) == D({'z': 1})
    assert update_in(D({}), ['z'], lambda x: x + 'ar', default='b', **kw) == D({'z': 'bar'})
    assert update_in(D({}), [0, 1], inc, default=-1, **kw) == D({0: D({1: 0})})
    assert update_in(D({}), [0, 1], str, default=100, **kw) == D({0: D({1: '100'})})
    assert update_in(D({'foo': 'bar', 1: 50}), ['d', 1, 0], str, 20, **kw) == D({'foo': 'bar', 1: 50, 'd': D({1: D({0: '20'})})})
    d = D({'x': 1})
    oldd = d
    update_in(d, ['x'], inc, **kw)
    assert d is oldd