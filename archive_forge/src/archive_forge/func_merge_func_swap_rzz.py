import itertools
from typing import cast, Any, Dict, List, Optional, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq_google import ops
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore
def merge_func_swap_rzz(ops1: Sequence['cirq.Operation'], ops2: Sequence['cirq.Operation']) -> bool:
    if not (len(ops1) == 1 and len(ops2) == 1):
        return False
    for op1, op2 in itertools.permutations([ops1[0], ops2[0]]):
        if op1.gate == cirq.SWAP and isinstance(op2.gate, cirq.ZZPowGate):
            return True
    return False