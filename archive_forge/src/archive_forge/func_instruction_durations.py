from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple
from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
@property
def instruction_durations(self):
    """Return the :class:`~qiskit.transpiler.InstructionDurations` object."""
    return self.target.durations()