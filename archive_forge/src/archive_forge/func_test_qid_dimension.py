from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_qid_dimension():
    assert ValidQubit('a').dimension == 2
    assert ValidQubit('a').with_dimension(3).dimension == 3
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQubit('a').with_dimension(0)
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQubit('a').with_dimension(-3)
    assert ValidQid('a', 3).dimension == 3
    assert ValidQid('a', 3).with_dimension(2).dimension == 2
    assert ValidQid('a', 3).with_dimension(4) == ValidQid('a', 4)
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQid('a', 3).with_dimension(0)
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQid('a', 3).with_dimension(-3)