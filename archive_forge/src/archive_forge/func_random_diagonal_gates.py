from itertools import combinations
from string import ascii_lowercase
from typing import Sequence, Dict, Tuple
import numpy as np
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def random_diagonal_gates(num_qubits: int, acquaintance_size: int) -> Dict[Tuple[cirq.Qid, ...], cirq.Gate]:
    return {Q: cirq.DiagonalGate(np.random.random(2 ** acquaintance_size)) for Q in combinations(cirq.LineQubit.range(num_qubits), acquaintance_size)}