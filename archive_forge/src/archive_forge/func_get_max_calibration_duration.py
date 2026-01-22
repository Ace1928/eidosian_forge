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
def get_max_calibration_duration(inst_map, cal_gate):
    """Return the time needed to allow for readout discrimination calibration pulses."""
    max_calibration_duration = 0
    for qubits in inst_map.qubits_with_instruction(cal_gate):
        cmd = inst_map.get(cal_gate, qubits, np.pi, 0, np.pi)
        max_calibration_duration = max(cmd.duration, max_calibration_duration)
    return max_calibration_duration