from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
def num_conseq_2qbit_gates(i):
    j = i
    while j < len(operations) and operations[j].gate.num_qubits() == 2:
        j += 1
    return j - i