from typing import cast
import pytest
import cirq
def test_transform_internal_nodes():
    operations = [cirq.GateOperation(cirq.testing.SingleQubitGate(), [cirq.LineQubit(2 * i)]) for i in range(10)]

    def skip_first(op):
        first = True
        for item in op:
            if not first:
                yield item
            first = False

    def skip_tree_freeze(root):
        return cirq.freeze_op_tree(cirq.transform_op_tree(root, iter_transformation=skip_first))
    assert skip_tree_freeze([[[]]]) == ()
    assert skip_tree_freeze([[[]], [[], []]]) == (((),),)
    assert skip_tree_freeze(operations[0]) == operations[0]
    assert skip_tree_freeze(operations) == tuple(operations[1:])
    assert skip_tree_freeze((operations[1:5], operations[0], operations[5:])) == (operations[0], tuple(operations[6:]))