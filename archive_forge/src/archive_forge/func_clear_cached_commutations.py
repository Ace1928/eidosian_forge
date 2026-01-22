from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def clear_cached_commutations(self):
    """Clears the dictionary holding cached commutations"""
    self._current_cache_entries = 0
    self._cache_miss = 0
    self._cache_hit = 0
    self._cached_commutations = {}