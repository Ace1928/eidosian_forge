import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
@staticmethod
def from_fsim_rz(theta: 'cirq.TParamVal', phi: 'cirq.TParamVal', rz_angles_before: Tuple['cirq.TParamVal', 'cirq.TParamVal'], rz_angles_after: Tuple['cirq.TParamVal', 'cirq.TParamVal']) -> 'PhasedFSimGate':
    """Creates PhasedFSimGate using an alternate parametrization.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                See class docstring above for details.
            phi: Controlled phase angle, in radians. See class docstring
                above for details.
            rz_angles_before: 2-tuple of phase angles to apply to each qubit
                before the core FSimGate. See class docstring for details.
            rz_angles_after: 2-tuple of phase angles to apply to each qubit
                after the core FSimGate. See class docstring for details.
        """
    b0, b1 = rz_angles_before
    a0, a1 = rz_angles_after
    gamma = (-b0 - b1 - a0 - a1) / 2.0
    zeta = (b0 - b1 + a0 - a1) / 2.0
    chi = (b0 - b1 - a0 + a1) / 2.0
    return PhasedFSimGate(theta, zeta, chi, gamma, phi)