import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
def generalized_amplitude_damp(p: float, gamma: float) -> GeneralizedAmplitudeDampingChannel:
    """Returns a GeneralizedAmplitudeDampingChannel with probabilities gamma and p.

    This channel evolves a density matrix via:

    $$
    \\rho \\rightarrow M_0 \\rho M_0^\\dagger + M_1 \\rho M_1^\\dagger
          + M_2 \\rho M_2^\\dagger + M_3 \\rho M_3^\\dagger
    $$

    With:

    $$
    \\begin{aligned}
    M_0 =& \\sqrt{p} \\begin{bmatrix}
                        1 & 0  \\\\
                        0 & \\sqrt{1 - \\gamma}
                   \\end{bmatrix}
    \\\\
    M_1 =& \\sqrt{p} \\begin{bmatrix}
                        0 & \\sqrt{\\gamma} \\\\
                        0 & 0
                   \\end{bmatrix}
    \\\\
    M_2 =& \\sqrt{1-p} \\begin{bmatrix}
                        \\sqrt{1-\\gamma} & 0 \\\\
                         0 & 1
                      \\end{bmatrix}
    \\\\
    M_3 =& \\sqrt{1-p} \\begin{bmatrix}
                         0 & 0 \\\\
                         \\sqrt{\\gamma} & 0
                     \\end{bmatrix}
    \\end{aligned}
    $$

    Args:
        gamma: the probability of the interaction being dissipative.
        p: the probability of the qubit and environment exchanging energy.

    Raises:
        ValueError: gamma or p is not a valid probability.
    """
    return GeneralizedAmplitudeDampingChannel(p, gamma)