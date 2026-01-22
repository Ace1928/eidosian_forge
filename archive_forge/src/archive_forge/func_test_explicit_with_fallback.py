import pytest
import cirq
def test_explicit_with_fallback():
    a2 = cirq.NamedQubit('a2')
    a10 = cirq.NamedQubit('a10')
    b = cirq.NamedQubit('b')
    q = cirq.QubitOrder.explicit([b], fallback=cirq.QubitOrder.DEFAULT)
    assert q.order_for([]) == (b,)
    assert q.order_for([b]) == (b,)
    assert q.order_for([b, a2]) == (b, a2)
    assert q.order_for([a2]) == (b, a2)
    assert q.order_for([a10, a2]) == (b, a2, a10)