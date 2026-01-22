import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_has_diagram():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(a, b))
    cirq.testing.assert_has_diagram(circuit, '\n0: ───@───\n      │\n1: ───X───\n')
    expected_error = "Circuit's text diagram differs from the desired diagram.\n\nDiagram of actual circuit:\n0: ───@───\n      │\n1: ───X───\n\nDesired text diagram:\n0: ───@───\n      │\n1: ───Z───\n\nHighlighted differences:\n0: ───@───\n      │\n1: ───█───\n\n"
    with pytest.raises(AssertionError) as ex_info:
        cirq.testing.assert_has_diagram(circuit, '\n0: ───@───\n      │\n1: ───Z───\n')
    assert expected_error in ex_info.value.args[0]