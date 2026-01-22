from __future__ import annotations
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
import numpy as np
from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.channels import ClassicalIOChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleComponent
def get_first_acquire_times(schedules):
    """Return a list of first acquire times for each schedule."""
    acquire_times = []
    for schedule in schedules:
        visited_channels = set()
        qubit_first_acquire_times: dict[int, int] = defaultdict(lambda: None)
        for time, inst in schedule.instructions:
            if isinstance(inst, instructions.Acquire) and inst.channel not in visited_channels:
                visited_channels.add(inst.channel)
                qubit_first_acquire_times[inst.channel.index] = time
        acquire_times.append(qubit_first_acquire_times)
    return acquire_times