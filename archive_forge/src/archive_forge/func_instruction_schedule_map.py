from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple
from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
@property
def instruction_schedule_map(self):
    """Return the :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions defined in this backend's target."""
    return self.target.instruction_schedule_map()