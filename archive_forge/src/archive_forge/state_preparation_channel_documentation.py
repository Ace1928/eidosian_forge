from typing import Any, Dict, Tuple, Iterable, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq.ops import raw_types
from cirq._compat import proper_repr
Returns the Kraus operator for this gate

        The Kraus Operator is |Psi><i| for all |i>, where |Psi> is the target state.
        This allows is to take any input state to the target state.
        The operator satisfies the completeness relation Sum(E^ E) = I.
        