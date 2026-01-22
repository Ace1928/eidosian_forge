from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate, CXGate
from .two_local import TwoLocal
Return the parameter bounds.

        Returns:
            The parameter bounds.
        