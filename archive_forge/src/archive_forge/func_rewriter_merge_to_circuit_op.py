from typing import List
import numpy as np
import pytest
import cirq
def rewriter_merge_to_circuit_op(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
    nonlocal component_id
    component_id = component_id + 1
    return op.with_tags(f'{component_id}')