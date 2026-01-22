import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_assert_url_to_circuit_returns_circuit():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', circuit=cirq.Circuit(cirq.X(cirq.LineQubit(0))))
    with pytest.raises(AssertionError, match='circuit differs'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', circuit=cirq.Circuit(cirq.Y(cirq.LineQubit(0))))