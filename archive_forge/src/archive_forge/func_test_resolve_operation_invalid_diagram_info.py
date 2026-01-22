import cirq
import cirq_web
import pytest
@pytest.mark.parametrize('custom_gate', [MockGateNoDiagramInfo, MockGateUnimplementedDiagramInfo])
def test_resolve_operation_invalid_diagram_info(custom_gate):
    mock_qubit = cirq.NamedQubit('mock')
    gate = custom_gate()
    operation = gate.on(mock_qubit)
    symbol_info = cirq_web.circuits.symbols.resolve_operation(operation, cirq_web.circuits.symbols.DEFAULT_SYMBOL_RESOLVERS)
    expected_labels = ['?']
    expected_colors = ['gray']
    assert symbol_info.labels == expected_labels
    assert symbol_info.colors == expected_colors