from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def get_physical_bits(self):
    """
        Returns the dictionary where the keys are physical (qu)bits and the
        values are virtual (qu)bits.
        """
    return self._p2v