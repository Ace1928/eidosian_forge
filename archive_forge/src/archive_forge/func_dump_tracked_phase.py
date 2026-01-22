from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np
from cirq import ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def dump_tracked_phase(qubits: Iterable[ops.Qid]) -> 'cirq.OP_TREE':
    """Zeroes qubit_phase entries by emitting Z gates."""
    for q in qubits:
        p, key = (qubit_phase[q], last_phased_xz_op[q])
        qubit_phase[q] = 0
        if not (key or single_qubit_decompositions.is_negligible_turn(p, atol)):
            yield (ops.Z(q) ** (p * 2))
        elif key:
            phased_xz_replacements[key] = phased_xz_replacements[key].with_z_exponent(p * 2)