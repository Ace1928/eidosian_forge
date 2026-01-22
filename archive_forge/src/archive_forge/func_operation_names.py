from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple
from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
@property
def operation_names(self) -> List[str]:
    """A list of instruction names that the backend supports."""
    return list(self.target.operation_names)