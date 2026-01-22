from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
def ch_duration(self, *channels: Channel) -> int:
    """Return duration of the supplied channels in this Instruction.

        Args:
            *channels: Supplied channels
        """
    return self.ch_stop_time(*channels)