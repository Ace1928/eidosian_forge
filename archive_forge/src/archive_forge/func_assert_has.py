from __future__ import annotations
import functools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Callable
from qiskit import circuit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.calibration_entries import (
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
def assert_has(self, instruction: str | circuit.instruction.Instruction, qubits: int | Iterable[int]) -> None:
    """Error if the given instruction is not defined.

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Raises:
            PulseError: If the instruction is not defined on the qubits.
        """
    instruction = _get_instruction_string(instruction)
    if not self.has(instruction, _to_tuple(qubits)):
        if instruction in self._map:
            raise PulseError("Operation '{inst}' exists, but is only defined for qubits {qubits}.".format(inst=instruction, qubits=self.qubits_with_instruction(instruction)))
        raise PulseError(f"Operation '{instruction}' is not defined for this system.")