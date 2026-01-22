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
@global_phase.setter
def global_phase(self, angle: ParameterValueType):
    """Set the phase of the current circuit scope.

        Args:
            angle (float, ParameterExpression): radians
        """
    global_phase_reference = (ParameterTable.GLOBAL_PHASE, None)
    if isinstance((previous := getattr(self, '_global_phase', None)), ParameterExpression):
        self._parameters = None
        self._parameter_table.discard_references(previous, global_phase_reference)
    if isinstance(angle, ParameterExpression) and angle.parameters:
        for parameter in angle.parameters:
            if parameter not in self._parameter_table:
                self._parameters = None
                self._parameter_table[parameter] = ParameterReferences(())
            self._parameter_table[parameter].add(global_phase_reference)
    else:
        angle = _normalize_global_phase(angle)
    if self._control_flow_scopes:
        self._control_flow_scopes[-1].global_phase = angle
    else:
        self._global_phase = angle