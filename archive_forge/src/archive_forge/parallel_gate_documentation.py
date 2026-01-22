from typing import AbstractSet, Union, Any, Optional, Tuple, TYPE_CHECKING, Dict
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType
Raises underlying gate to a power, applying same number of copies.

        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5) x k  or  (G x k) ** 1.5

        Args:
            exponent: The amount to scale the gate's effect by.

        Returns:
            ParallelGate with same num_copies with the scaled underlying gate.
        