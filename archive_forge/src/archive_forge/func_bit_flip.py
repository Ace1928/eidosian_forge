import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
def bit_flip(p: Optional[float]=None) -> Union[common_gates.XPowGate, BitFlipChannel]:
    """Construct a BitFlipChannel that flips a qubit state with probability p.

    If p is None, this returns a guaranteed flip in the form of an X operation.

    This channel evolves a density matrix via

    $$
    \\rho \\rightarrow M_0 \\rho M_0^\\dagger + M_1 \\rho M_1^\\dagger
    $$

    With

    $$
    \\begin{aligned}
    M_0 =& \\sqrt{1-p} \\begin{bmatrix}
                        1 & 0 \\\\
                        0 & 1
                   \\end{bmatrix}
    \\\\
    M_1 =& \\sqrt{p} \\begin{bmatrix}
                        0 & 1 \\\\
                        1 & 0
                     \\end{bmatrix}
    \\end{aligned}
    $$

    Args:
        p: the probability of a bit flip.

    Raises:
        ValueError: if p is not a valid probability.
    """
    if p is None:
        return pauli_gates.X
    return _bit_flip(p)