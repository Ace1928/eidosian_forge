from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('use_circuit_op', [True, False])
def test_gateset_validate(use_circuit_op):

    def optree_and_circuit(optree):
        yield optree
        yield cirq.Circuit(optree)

    def get_ops(use_circuit_op):
        q = cirq.LineQubit.range(3)
        yield [CustomX(q[0]).with_tags('custom tags'), CustomX(q[1]) ** 2, CustomX(q[2]) ** 3]
        yield [CustomX(q[0]) ** 0.5, cirq.testing.TwoQubitGate()(*q[:2])]
        if use_circuit_op:
            circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit(get_ops(False)), repetitions=10).with_tags('circuit op tags')
            recursive_circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit([circuit_op, CustomX(q[2]) ** 0.5]), repetitions=10, qubit_map={q[0]: q[1], q[1]: q[2], q[2]: q[0]})
            yield [circuit_op, recursive_circuit_op]

    def assert_validate_and_contains_consistent(gateset, op_tree, result):
        assert all((op in gateset for op in cirq.flatten_to_ops(op_tree))) is result
        for item in optree_and_circuit(op_tree):
            assert gateset.validate(item) is result
    op_tree = [*get_ops(use_circuit_op)]
    assert_validate_and_contains_consistent(gateset.with_params(unroll_circuit_op=use_circuit_op), op_tree, True)
    if use_circuit_op:
        assert_validate_and_contains_consistent(gateset.with_params(unroll_circuit_op=False), op_tree, False)