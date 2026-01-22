from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, arithmetic_gates
Prepares a uniform superposition over first $n$ basis states using $O(log(n))$ T-gates.

    Performs a single round of amplitude amplification and prepares a uniform superposition over
    the first $n$ basis states $|0>, |1>, ..., |n - 1>$. The expected T-complexity should be
    $10 * log(L) + 2 * K$ T-gates and $2$ single qubit rotation gates, where $n = L * 2^K$.

    However, the current T-complexity is $12 * log(L)$ T-gates and $2 + 2 * (K + log(L))$ rotations
    because of two open issues:
        - https://github.com/quantumlib/cirq-qubitization/issues/233 and
        - https://github.com/quantumlib/cirq-qubitization/issues/235

    Args:
        n: The gate prepares a uniform superposition over first $n$ basis states.
        cv: Control values for each control qubit. If specified, a controlled version
            of the gate is constructed.

    References:
        See Fig 12 of https://arxiv.org/abs/1805.03662 for more details.
    