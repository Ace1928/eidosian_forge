import sympy
import cirq
from cirq.study import flatten_expressions
def test_flatten_circuit():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.X(qubit) ** a, cirq.X(qubit) ** (1 + a / 2))
    c_flat, expr_map = cirq.flatten(circuit)
    c_expected = cirq.Circuit(cirq.X(qubit) ** a, cirq.X(qubit) ** sympy.Symbol('<a/2 + 1>'))
    assert c_flat == c_expected
    assert isinstance(expr_map, cirq.ExpressionMap)
    assert expr_map == {a: a, 1 + a / 2: sympy.Symbol('<a/2 + 1>')}