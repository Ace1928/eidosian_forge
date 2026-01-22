from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (

    Synthesize linear reversible circuit for linear nearest-neighbor architectures using
    Kutin, Moulton, Smithline method.

    Synthesis algorithm for linear reversible circuits from [1], section 7.
    This algorithm synthesizes any linear reversible circuit of :math:`n` qubits over
    a linear nearest-neighbor architecture using CX gates with depth at most :math:`5n`.

    Args:
        mat: A boolean invertible matrix.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``mat`` is not invertible.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
    