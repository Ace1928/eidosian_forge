from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def get_mergeable_ops(self, op: 'cirq.Operation', op_qs: Set['cirq.Qid']) -> Tuple[int, List['cirq.Operation']]:
    idx = max([self.qubit_indexes[q][-1] for q in op_qs], default=-1)
    idx = max([idx] + [self.mkey_indexes[ckey][-1] for ckey in protocols.control_keys(op)])
    idx = max([idx] + [self.ckey_indexes[mkey][-1] for mkey in protocols.measurement_key_objs(op)])
    if idx == -1:
        return (idx, [])
    return (idx, [left_op for left_op in self.ops_by_index[idx] if not op_qs.isdisjoint(left_op.qubits)])