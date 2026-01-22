import os
import pytest
from monty.collections import AttrDict, FrozenAttrDict, Namespace, frozendict, tree
def test_frozen_attrdict(self):
    d = FrozenAttrDict({'hello': 'world', 1: 2})
    assert d['hello'] == 'world'
    assert d.hello == 'world'
    with pytest.raises(KeyError):
        d['updating'] == 2
    with pytest.raises(KeyError):
        d['foo'] = 'bar'
    with pytest.raises(KeyError):
        d.foo = 'bar'
    with pytest.raises(KeyError):
        d.hello = 'new'