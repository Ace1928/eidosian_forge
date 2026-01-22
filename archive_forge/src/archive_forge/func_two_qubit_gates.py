import abc
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List, Set, Type
from cirq import _compat, ops, devices
from cirq.devices import noise_utils
@classmethod
def two_qubit_gates(cls) -> Set[Type[ops.Gate]]:
    """Returns the set of all two-qubit gates this class supports."""
    return cls.symmetric_two_qubit_gates() | cls.asymmetric_two_qubit_gates()