from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
Returns true if this is contained within op_id, but not equal to it.

        If this returns true, (x in self) implies (x in op_id), but the reverse
        implication does not hold. op_id must be more general than self (either
        by accepting any qubits or having a more general gate type) for this
        to return true.
        