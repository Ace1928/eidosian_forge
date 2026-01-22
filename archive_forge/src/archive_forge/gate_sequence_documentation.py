from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
Compute the dot-product with another gate sequence.

        Args:
            other: The other gate sequence.

        Returns:
            The dot-product as gate sequence.
        