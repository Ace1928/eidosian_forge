from typing import Union, List
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
Create new HLF circuit.

        Args:
            adjacency_matrix: a symmetric n-by-n list of 0-1 lists.
                n will be the number of qubits.

        Raises:
            CircuitError: If A is not symmetric.
        