from typing import Union, Optional
import math
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.s import SGate, SdgGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.states.statevector import Statevector  # pylint: disable=cyclic-import
def validate_parameter(self, parameter):
    """StatePreparation instruction parameter can be str, int, float, and complex."""
    if isinstance(parameter, str):
        if parameter in ['0', '1', '+', '-', 'l', 'r']:
            return parameter
        raise CircuitError('invalid param label {} for instruction {}. Label should be 0, 1, +, -, l, or r '.format(type(parameter), self.name))
    if isinstance(parameter, (int, float, complex)):
        return complex(parameter)
    elif isinstance(parameter, np.number):
        return complex(parameter.item())
    else:
        raise CircuitError(f'invalid param type {type(parameter)} for instruction  {self.name}')