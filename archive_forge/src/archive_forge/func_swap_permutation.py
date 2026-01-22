from __future__ import annotations
from collections.abc import Iterable
from typing import TypeVar, MutableMapping
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from .types import Swap, PermutationCircuit
def swap_permutation(swaps: Iterable[Iterable[Swap[_K]]], mapping: MutableMapping[_K, _V], allow_missing_keys: bool=False) -> None:
    """Given a circuit of swaps, apply them to the permutation (in-place).

    Args:
      swaps: param mapping: A mapping of Keys to Values, where the Keys are being swapped.
      mapping: The permutation to have swaps applied to.
      allow_missing_keys: Whether to allow swaps of missing keys in mapping.
    """
    for swap_step in swaps:
        for sw1, sw2 in swap_step:
            val1: _V | None = None
            val2: _V | None = None
            if allow_missing_keys:
                val1 = mapping.pop(sw1, None)
                val2 = mapping.pop(sw2, None)
            else:
                val1, val2 = (mapping.pop(sw1), mapping.pop(sw2))
            if val1 is not None:
                mapping[sw2] = val1
            if val2 is not None:
                mapping[sw1] = val2