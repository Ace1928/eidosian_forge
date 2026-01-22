import json
import logging
import warnings
from json import JSONEncoder
from typing import (
from pyquil.experiment._calibration import CalibrationMethod
from pyquil.experiment._memory import (
from pyquil.experiment._program import (
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, _OneQState, TensorProductState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import Program
from pyquil.quilbase import Reset, ResetQubit
def get_meas_registers(self, qubits: Optional[Sequence[int]]=None) -> List[int]:
    """
        Return the sorted list of memory registers corresponding to the list of qubits provided.
        If no qubits are provided, just returns the list of numbers from 0 to n-1 where n is the
        number of qubits resulting from the ``get_meas_qubits`` method.
        """
    meas_qubits = self.get_meas_qubits()
    if qubits is None:
        return list(range(len(meas_qubits)))
    meas_registers = []
    for q in qubits:
        meas_registers.append(meas_qubits.index(q))
    return sorted(meas_registers)