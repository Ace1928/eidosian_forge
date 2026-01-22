import pytest
import cirq
def test_qid_shape():

    class ShapeObj:

        def _qid_shape_(self):
            return (1, 2, 3)

    class NumObj:

        def _num_qubits_(self):
            return 2

    class NotImplShape:

        def _qid_shape_(self):
            return NotImplemented

    class NotImplNum:

        def _num_qubits_(self):
            return NotImplemented

    class NotImplBoth:

        def _num_qubits_(self):
            return NotImplemented

        def _qid_shape_(self):
            return NotImplemented

    class NoProtocol:
        pass
    assert cirq.qid_shape(ShapeObj()) == (1, 2, 3)
    assert cirq.num_qubits(ShapeObj()) == 3
    assert cirq.qid_shape(NumObj()) == (2, 2)
    assert cirq.num_qubits(NumObj()) == 2
    with pytest.raises(TypeError, match='_qid_shape_.*NotImplemented'):
        cirq.qid_shape(NotImplShape())
    with pytest.raises(TypeError, match='_qid_shape_.*NotImplemented'):
        cirq.num_qubits(NotImplShape())
    with pytest.raises(TypeError, match='_num_qubits_.*NotImplemented'):
        cirq.qid_shape(NotImplNum())
    with pytest.raises(TypeError, match='_num_qubits_.*NotImplemented'):
        cirq.num_qubits(NotImplNum())
    with pytest.raises(TypeError, match='_qid_shape_.*NotImplemented'):
        cirq.qid_shape(NotImplBoth())
    with pytest.raises(TypeError, match='_num_qubits_.*NotImplemented'):
        cirq.num_qubits(NotImplBoth())
    with pytest.raises(TypeError):
        cirq.qid_shape(NoProtocol())
    with pytest.raises(TypeError):
        cirq.num_qubits(NoProtocol())
    assert cirq.qid_shape(cirq.LineQid.for_qid_shape((1, 2, 3))) == (1, 2, 3)
    assert cirq.num_qubits(cirq.LineQid.for_qid_shape((1, 2, 3))) == 3