from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def unroll_circuit_op(circuit: CIRCUIT_TYPE, *, deep: bool=False, tags_to_check: Optional[Sequence[Hashable]]=(MAPPED_CIRCUIT_OP_TAG,)) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s while preserving the moment structure.

    Each moment containing a matching circuit operation is expanded into a list of moments with the
    unrolled operations, hence preserving the original moment structure.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded in a moment preserving way.
    """

    def map_func(m: circuits.Moment, _: int):
        to_zip: List['cirq.AbstractCircuit'] = []
        for op in m:
            op_untagged = op.untagged
            if isinstance(op_untagged, circuits.CircuitOperation):
                if deep:
                    op_untagged = op_untagged.replace(circuit=unroll_circuit_op(op_untagged.circuit, deep=deep, tags_to_check=tags_to_check))
                to_zip.append(op_untagged.mapped_circuit() if tags_to_check is None or set(tags_to_check).intersection(op.tags) else circuits.Circuit(op_untagged.with_tags(*op.tags)))
            else:
                to_zip.append(circuits.Circuit(op))
        return circuits.Circuit.zip(*to_zip).moments
    return map_moments(circuit, map_func)