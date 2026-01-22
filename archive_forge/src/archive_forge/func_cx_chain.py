from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate
from ..n_local.n_local import NLocal
def cx_chain(circuit, inverse=False):
    num_cx = len(indices) - 1
    for i in reversed(range(num_cx)) if inverse else range(num_cx):
        circuit.cx(indices[i], indices[i + 1])