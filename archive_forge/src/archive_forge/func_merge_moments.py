from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def merge_moments(circuit: CIRCUIT_TYPE, merge_func: Callable[[circuits.Moment, circuits.Moment], Optional[circuits.Moment]], *, tags_to_ignore: Sequence[Hashable]=(), deep: bool=False) -> CIRCUIT_TYPE:
    """Merges adjacent moments, one by one from left to right, by calling `merge_func(m1, m2)`.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        merge_func: Callable to determine whether two adjacent moments in the circuit should be
            merged. If the moments can be merged, the callable should return the merged moment,
            else None.
        tags_to_ignore: Tagged circuit operations marked with any of `tags_to_ignore` will be
            ignored when recursively applying the transformer primitive to sub-circuits, given
            deep=True.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.

    Returns:
        Copy of input circuit with merged moments.
    """
    if not circuit:
        return circuit
    if deep:
        circuit = map_operations(circuit, lambda op, _: op.untagged.replace(circuit=merge_moments(op.untagged.circuit, merge_func, deep=deep)).with_tags(*op.tags) if isinstance(op.untagged, circuits.CircuitOperation) else op, tags_to_ignore=tags_to_ignore)
    merged_moments: List[circuits.Moment] = [circuit[0]]
    for current_moment in circuit[1:]:
        merged_moment = merge_func(merged_moments[-1], current_moment)
        if merged_moment is None:
            merged_moments.append(current_moment)
        else:
            merged_moments[-1] = merged_moment
    return _create_target_circuit_type(merged_moments, circuit)