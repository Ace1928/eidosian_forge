from __future__ import annotations
import copy
import multiprocessing as mp
import typing
from collections import OrderedDict, defaultdict, namedtuple
from typing import (
import numpy as np
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from . import _classical_resource_map
from ._utils import sort_parameters
from .controlflow.builder import CircuitScopeInterface, ControlFlowBuilderBlock
from .controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
from .controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
from .controlflow.for_loop import ForLoopOp, ForLoopContext
from .controlflow.if_else import IfElseOp, IfContext
from .controlflow.switch_case import SwitchCaseOp, SwitchContext
from .controlflow.while_loop import WhileLoopOp, WhileLoopContext
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
def qubit_stop_time(self, *qubits: Union[Qubit, int]) -> float:
    """Return the stop time of the last instruction, excluding delays, over the supplied qubits.
        Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the stop time of the last instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
    if self.duration is None:
        for instruction in self._data:
            if not isinstance(instruction.operation, Delay):
                raise CircuitError('qubit_stop_time undefined. Circuit must be scheduled first.')
        return 0
    qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]
    stops = {q: self.duration for q in qubits}
    dones = {q: False for q in qubits}
    for instruction in reversed(self._data):
        for q in qubits:
            if q in instruction.qubits:
                if isinstance(instruction.operation, Delay):
                    if not dones[q]:
                        stops[q] -= instruction.operation.duration
                else:
                    dones[q] = True
        if len(qubits) == len([done for done in dones.values() if done]):
            return max((stop for stop in stops.values()))
    return 0