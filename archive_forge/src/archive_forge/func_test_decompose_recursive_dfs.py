import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
@pytest.mark.parametrize('with_context', [True, False])
def test_decompose_recursive_dfs(with_context: bool):
    expected_calls = [mock.call.qalloc(True), mock.call.qalloc(False), mock.call.qfree(False), mock.call.qfree(True)]
    mock_qm = mock.Mock(spec=cirq.QubitManager)
    context_qm = mock.Mock(spec=cirq.QubitManager)
    gate = RecursiveDecompose(mock_qm=mock_qm, with_context=with_context)
    q = cirq.LineQubit.range(3)
    gate_op = gate.on(*q[:2])
    tagged_op = gate_op.with_tags('custom tag')
    controlled_op = gate_op.controlled_by(q[2])
    classically_controlled_op = gate_op.with_classical_controls('key')
    moment = cirq.Moment(gate_op)
    circuit = cirq.Circuit(moment)
    for val in [gate_op, tagged_op, controlled_op, classically_controlled_op, moment, circuit]:
        mock_qm.reset_mock()
        _ = cirq.decompose(val, context=cirq.DecompositionContext(qubit_manager=mock_qm))
        assert mock_qm.method_calls == expected_calls
        mock_qm.reset_mock()
        context_qm.reset_mock()
        _ = cirq.decompose(val, context=cirq.DecompositionContext(context_qm))
        assert context_qm.method_calls == expected_calls if with_context else mock_qm.method_calls == expected_calls