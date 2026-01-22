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
def with_instruction_types(types: Iterable[abc.ABCMeta] | abc.ABCMeta) -> Callable:
    """Instruction type filter generator.

    Args:
        types: List of instruction types to filter.

    Returns:
        A callback function to filter instructions.
    """
    types = _if_scalar_cast_to_list(types)

    @singledispatch
    def instruction_filter(time_inst) -> bool:
        """A catch-TypeError function which will only get called if none of the other decorated
        functions, namely handle_numpyndarray() and handle_instruction(), handle the type passed.
        """
        raise TypeError(f"Type '{type(time_inst)}' is not valid data format as an input to instruction_filter.")

    @instruction_filter.register
    def handle_numpyndarray(time_inst: np.ndarray) -> bool:
        """Filter instruction.

        Args:
            time_inst (numpy.ndarray([int, Instruction])): Time

        Returns:
            If instruction matches with condition.
        """
        return isinstance(time_inst[1], tuple(types))

    @instruction_filter.register
    def handle_instruction(inst: Instruction) -> bool:
        """Filter instruction.

        Args:
            inst: Instruction

        Returns:
            If instruction matches with condition.
        """
        return isinstance(inst, tuple(types))
    return instruction_filter