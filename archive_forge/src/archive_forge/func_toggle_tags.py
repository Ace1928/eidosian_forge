from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def toggle_tags(circuit: CIRCUIT_TYPE, tags: Sequence[Hashable], *, deep: bool=False):
    """Toggles tags applied on each operation in the circuit, via `op.tags ^= tags`

    For every operations `op` in the input circuit, the tags on `op` are replaced by a symmetric
    difference of `op.tags` and `tags` -- this is useful in scenarios where you mark a small subset
    of operations with a specific tag and then toggle the set of marked operations s.t. every
    marked operation is now unmarked and vice versa.

    Often used in transformer workflows to apply a transformer on a small subset of operations.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        tags: Sequence of tags s.t. `op.tags ^= tags` is done for every operation `op` in circuit.
        deep: If true, tags will be recursively toggled for operations in circuits wrapped inside
            any circuit operations contained within `circuit`.

    Returns:
        Copy of transformed input circuit with operation sets marked with `tags` toggled.
    """
    tags_to_xor = set(tags)

    def map_func(op: 'cirq.Operation', _) -> 'cirq.Operation':
        return op if deep and isinstance(op, circuits.CircuitOperation) else op.untagged.with_tags(*set(op.tags) ^ tags_to_xor)
    return map_operations(circuit, map_func, deep=deep)