import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict
from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType
def phased_pauli_product(self, other: Union['cirq.Pauli', 'identity.IdentityGate']) -> Tuple[complex, Union['cirq.Pauli', 'identity.IdentityGate']]:
    if self == other:
        return (1, identity.I)
    if other is identity.I:
        return (1, self)
    return (1j ** cast(Pauli, other).relative_index(self), self.third(cast(Pauli, other)))