from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def project_Z(self, q, z):
    """Applies a Z projector on the q'th qubit.

        Returns: a normalized state with Z_q |psi> = z |psi>
        """
    t = self.s.copy()
    u = self.G[q, :] & self.v ^ self.s
    delta = (2 * sum(self.G[q, :] & ~self.v & self.s) + 2 * z) % 4
    if np.all(t == u):
        self.omega /= np.sqrt(2)
    self.update_sum(t, u, delta=delta)