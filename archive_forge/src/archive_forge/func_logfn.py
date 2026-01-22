from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def logfn(x):
    return -x * np.log(x) / np.log(base)