from typing import Optional
import warnings
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction
from ..blueprintcircuit import BlueprintCircuit
@do_swaps.setter
def do_swaps(self, do_swaps: bool) -> None:
    """Specify whether to do the final swaps of the QFT circuit or not.

        Args:
            do_swaps: If True, the final swaps are applied, if False not.
        """
    if do_swaps != self._do_swaps:
        self._invalidate()
        self._do_swaps = do_swaps