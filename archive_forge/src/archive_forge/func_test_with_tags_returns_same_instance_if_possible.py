from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_with_tags_returns_same_instance_if_possible():
    untagged = cirq.X(cirq.GridQubit(1, 1))
    assert untagged.with_tags() is untagged
    tagged = untagged.with_tags('foo')
    assert tagged.with_tags() is tagged