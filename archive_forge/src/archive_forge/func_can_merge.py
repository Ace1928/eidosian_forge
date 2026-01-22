from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def can_merge(ops1: List['cirq.Operation'], ops2: List['cirq.Operation']) -> bool:
    """Artificial example where a CZ will absorb any merge-able operation."""
    return any((o.gate == cirq.CZ for op_list in [ops1, ops2] for o in op_list))