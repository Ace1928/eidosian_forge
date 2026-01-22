import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_circuit_operations():
    q = cirq.NamedQubit('q')
    circuit = cirq.FrozenCircuit(cirq.T(q), cirq.X(q) ** 0.5, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert cirq_ft.t_complexity(cirq.CircuitOperation(circuit)) == cirq_ft.TComplexity(clifford=2, rotations=1, t=1)
    assert cirq_ft.t_complexity(cirq.CircuitOperation(circuit, repetitions=10)) == cirq_ft.TComplexity(clifford=20, rotations=10, t=10)
    circuit = cirq.FrozenCircuit(cirq.T(q) ** (-1), cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert cirq_ft.t_complexity(cirq.CircuitOperation(circuit)) == cirq_ft.TComplexity(clifford=1, rotations=1, t=1)
    assert cirq_ft.t_complexity(cirq.CircuitOperation(circuit, repetitions=3)) == cirq_ft.TComplexity(clifford=3, rotations=3, t=3)