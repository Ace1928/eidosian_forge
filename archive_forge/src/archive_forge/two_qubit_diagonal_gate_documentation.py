from typing import AbstractSet, Any, Dict, Tuple, Optional, Sequence, TYPE_CHECKING
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, common_gates
A two qubit gate with only diagonal elements.

        This gate's off-diagonal elements are zero and its on-diagonal
        elements are all phases.

        Args:
            diag_angles_radians: The list of angles on the diagonal in radians.
                If these values are $(x_0, x_1, \ldots , x_3)$ then the unitary
                has diagonal values $(e^{i x_0}, e^{i x_1}, \ldots, e^{i x_3})$.
        