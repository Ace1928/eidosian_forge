from typing import cast
import pytest
import cirq
def test_freeze_op_tree():
    operations = [cirq.GateOperation(cirq.testing.SingleQubitGate(), [cirq.NamedQubit(str(i))]) for i in range(10)]
    assert cirq.freeze_op_tree([[[]]]) == (((),),)
    assert cirq.freeze_op_tree(operations[0]) == operations[0]
    assert cirq.freeze_op_tree(operations) == tuple(operations)
    assert cirq.freeze_op_tree((operations[0], (operations[i] for i in range(1, 5)), operations[5:])) == (operations[0], tuple(operations[1:5]), tuple(operations[5:]))
    with pytest.raises(TypeError):
        cirq.freeze_op_tree(None)
    with pytest.raises(TypeError):
        cirq.freeze_op_tree(5)
    with pytest.raises(TypeError):
        _ = cirq.freeze_op_tree([operations[0], (4,)])