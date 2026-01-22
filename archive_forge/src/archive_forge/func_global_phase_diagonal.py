from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
def global_phase_diagonal(par, wires, controls, control_values):
    """Returns the diagonal of a C(GlobalPhase) operator."""
    diag = np.ones(2 ** len(wires), dtype=np.complex128)
    controls = np.array(controls)
    control_values = np.array(control_values)
    ind = np.argsort(controls)
    controls = controls[ind[-1::-1]]
    control_values = control_values[ind[-1::-1]]
    idx = np.arange(2 ** len(wires), dtype=np.int64).reshape([2 for _ in wires])
    for c, w in zip(control_values, controls):
        idx = np.take(idx, np.array(int(c)), w)
    diag[idx.ravel()] = np.exp(-1j * par)
    return diag