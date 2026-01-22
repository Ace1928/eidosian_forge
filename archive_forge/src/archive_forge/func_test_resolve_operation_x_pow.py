import cirq
import cirq_web
import pytest
def test_resolve_operation_x_pow():
    mock_qubit = cirq.NamedQubit('mock')
    operation = cirq.X(mock_qubit) ** 0.5
    symbol_info = cirq_web.circuits.symbols.resolve_operation(operation, cirq_web.circuits.symbols.DEFAULT_SYMBOL_RESOLVERS)
    expected_labels = ['X^0.5']
    expected_colors = ['black']
    assert symbol_info.labels == expected_labels
    assert symbol_info.colors == expected_colors