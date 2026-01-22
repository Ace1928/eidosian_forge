import pytest
import cirq
from cirq import quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.composite_cell import _iterator_to_iterable
def test_iterator_to_iterable():
    k = 0

    def counter():
        nonlocal k
        k += 1
        return k - 1
    k = 0
    generator = (counter() for _ in range(10))
    assert k == 0
    assert list(generator) == list(range(10))
    assert k == 10
    assert list(generator) == []
    assert k == 10
    k = 0
    generator = _iterator_to_iterable((counter() for _ in range(10)))
    assert k == 0
    assert list(generator) == list(range(10))
    assert k == 10
    assert list(generator) == list(range(10))
    assert k == 10
    k = 0
    generator = _iterator_to_iterable((counter() for _ in range(10)))
    iter1 = iter(generator)
    iter2 = iter(generator)
    assert k == 0
    assert next(iter1) == 0
    assert k == 1
    assert next(iter1) == 1
    assert k == 2
    assert next(iter2) == 0
    assert k == 2
    assert next(iter2) == 1
    assert k == 2
    assert next(iter2) == 2
    assert k == 3
    assert list(iter1) == list(range(2, 10))
    assert k == 10
    assert list(iter2) == list(range(3, 10))
    assert k == 10