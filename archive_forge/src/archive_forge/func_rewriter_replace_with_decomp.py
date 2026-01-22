from typing import List
import numpy as np
import pytest
import cirq
def rewriter_replace_with_decomp(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
    nonlocal component_id
    component_id = component_id + 1
    tag = f'{component_id}'
    if len(op.qubits) == 1:
        return [cirq.T(op.qubits[0]).with_tags(tag)]
    one_layer = [op.with_tags(tag) for op in cirq.T.on_each(*op.qubits)]
    two_layer = [cirq.SQRT_ISWAP(*op.qubits).with_tags(tag)]
    return [one_layer, two_layer, one_layer]