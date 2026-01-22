from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.exceptions import CircuitError
from ..boolean_logic import OR
from ..blueprintcircuit import BlueprintCircuit
@geq.setter
def geq(self, geq: bool) -> None:
    """Set whether the comparator compares greater or less equal.

        Args:
            geq: If True, the comparator compares ``>=``, if False ``<``.
        """
    if geq != self._geq:
        self._invalidate()
        self._geq = geq