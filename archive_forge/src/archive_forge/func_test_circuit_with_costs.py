import cirq
import cirq_ft
import cirq_ft.infra.testing as cq_testing
import IPython.display
import ipywidgets
import pytest
from cirq_ft.infra.jupyter_tools import display_gate_and_compilation, svg_circuit
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_circuit_with_costs():
    g = cq_testing.GateHelper(cirq_ft.And(cv=(1, 1, 1)))
    circuit = cirq_ft.infra.jupyter_tools.circuit_with_costs(g.circuit)
    expected_circuit = cirq.Circuit(g.operation.with_tags('t:8,r:0'))
    cirq.testing.assert_same_circuits(circuit, expected_circuit)