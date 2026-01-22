from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
@property
def stop_time(self) -> int:
    """Relative end time of this instruction."""
    return self.duration