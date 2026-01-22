from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
def is_proper_subtype_of(self, op_id: 'OpIdentifier'):
    """Returns true if this is contained within op_id, but not equal to it.

        If this returns true, (x in self) implies (x in op_id), but the reverse
        implication does not hold. op_id must be more general than self (either
        by accepting any qubits or having a more general gate type) for this
        to return true.
        """
    more_specific_qubits = self.qubits and (not op_id.qubits)
    more_specific_gate = self.gate_type != op_id.gate_type and issubclass(self.gate_type, op_id.gate_type)
    if more_specific_qubits:
        return more_specific_gate or self.gate_type == op_id.gate_type
    elif more_specific_gate:
        return more_specific_qubits or self.qubits == op_id.qubits
    else:
        return False