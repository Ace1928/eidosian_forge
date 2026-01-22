from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from .p import PhaseGate
Return inverted CCZ gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CCZGate: inverse gate (self-inverse).
        