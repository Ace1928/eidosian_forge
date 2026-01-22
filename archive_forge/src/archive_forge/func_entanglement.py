from __future__ import annotations
import typing
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.exceptions import QiskitError
from ..blueprintcircuit import BlueprintCircuit
@entanglement.setter
def entanglement(self, entanglement: str | list[str] | list[list[str]] | list[int] | list[list[int]] | list[list[list[int]]] | list[list[list[list[int]]]] | Callable[[int], str] | Callable[[int], list[list[int]]] | None) -> None:
    """Set the entanglement strategy.

        Args:
            entanglement: The entanglement strategy. See :meth:`get_entangler_map` for more detail
                on the supported formats.
        """
    self._invalidate()
    self._entanglement = entanglement