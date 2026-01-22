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
@singledispatch
def channel_filter(time_inst):
    """A catch-TypeError function which will only get called if none of the other decorated
        functions, namely handle_numpyndarray() and handle_instruction(), handle the type passed.
        """
    raise TypeError(f"Type '{type(time_inst)}' is not valid data format as an input to channel_filter.")