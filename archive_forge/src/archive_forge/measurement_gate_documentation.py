from typing import (
import numpy as np
from cirq import _compat, protocols, value
from cirq.ops import raw_types
Returns the invert mask for all qubits.

        If the user supplies a partial invert_mask, this returns that mask
        padded by False.

        Similarly if no invert_mask is supplies this returns a tuple
        of size equal to the number of qubits with all entries False.
        