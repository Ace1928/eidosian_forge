from __future__ import annotations
import abc
from functools import singledispatch
from collections.abc import Iterable
from typing import Callable, Any, List
import numpy as np
from qiskit.pulse import Schedule, ScheduleBlock, Instruction
from qiskit.pulse.channels import Channel
from qiskit.pulse.schedule import Interval
from qiskit.pulse.exceptions import PulseError
@instruction_filter.register
def handle_numpyndarray(time_inst: np.ndarray) -> bool:
    """Filter instruction.

        Args:
            time_inst (numpy.ndarray([int, Instruction])): Time

        Returns:
            If instruction matches with condition.
        """
    return isinstance(time_inst[1], tuple(types))