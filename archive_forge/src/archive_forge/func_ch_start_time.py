from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
def ch_start_time(self, *channels: Channel) -> int:
    """Return minimum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
    return 0