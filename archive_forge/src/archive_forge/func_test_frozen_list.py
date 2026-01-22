from typing import Dict, List
import catalogue
import pytest
from pytest import raises
from confection import Config, SimpleFrozenDict, SimpleFrozenList, registry
def test_frozen_list():
    frozen = SimpleFrozenList(range(10))
    for k in range(10):
        assert frozen[k] == k
    with raises(NotImplementedError, match='frozen list'):
        frozen.append(5)
    with raises(NotImplementedError, match='frozen list'):
        frozen.reverse()
    with raises(NotImplementedError, match='frozen list'):
        frozen.pop(0)