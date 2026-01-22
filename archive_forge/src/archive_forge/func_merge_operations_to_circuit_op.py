from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def merge_operations_to_circuit_op(circuit: CIRCUIT_TYPE, can_merge: Callable[[Sequence['cirq.Operation'], Sequence['cirq.Operation']], bool], *, tags_to_ignore: Sequence[Hashable]=(), merged_circuit_op_tag: str='Merged connected component', deep: bool=False) -> CIRCUIT_TYPE:
    """Merges connected components of operations and wraps each component into a circuit operation.

    Uses `cirq.merge_operations` to identify connected components of operations. Moment structure
    is preserved for operations that do not participate in merging. For merged operations, the
    newly created circuit operations are constructed by inserting operations using EARLIEST
    strategy.
    If you need more control on moment structure of newly created circuit operations, consider
    using `cirq.merge_operations` directly with a custom `merge_func`.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        can_merge: Callable to determine whether a new operation `right_op` can be merged into an
            existing connected component of operations `left_ops` based on boolen returned by
            `can_merge(left_ops, right_op)`.
        tags_to_ignore: Tagged operations marked any of `tags_to_ignore` will not be considered as
            potential candidates for any connected component.
        merged_circuit_op_tag: Tag to be applied on circuit operations wrapping valid connected
            components.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.

    Returns:
        Copy of input circuit with valid connected components wrapped in tagged circuit operations.
    """

    def merge_func(op1: 'cirq.Operation', op2: 'cirq.Operation') -> Optional['cirq.Operation']:

        def get_ops(op: 'cirq.Operation'):
            op_untagged = op.untagged
            return [*op_untagged.circuit.all_operations()] if isinstance(op_untagged, circuits.CircuitOperation) and merged_circuit_op_tag in op.tags else [op]
        left_ops, right_ops = (get_ops(op1), get_ops(op2))
        if not can_merge(left_ops, right_ops):
            return None
        return circuits.CircuitOperation(circuits.FrozenCircuit(left_ops, right_ops)).with_tags(merged_circuit_op_tag)
    return merge_operations(circuit, merge_func, tags_to_ignore=tags_to_ignore, deep=deep)