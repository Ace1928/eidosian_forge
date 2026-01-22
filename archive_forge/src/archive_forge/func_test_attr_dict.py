import os
import pytest
from monty.collections import AttrDict, FrozenAttrDict, Namespace, frozendict, tree
def test_attr_dict(self):
    d = AttrDict(foo=1, bar=2)
    assert d.bar == 2
    assert d['foo'] == d.foo
    d.bar = 'hello'
    assert d['bar'] == 'hello'