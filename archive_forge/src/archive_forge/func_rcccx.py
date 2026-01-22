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
def rcccx(self, control_qubit1: QubitSpecifier, control_qubit2: QubitSpecifier, control_qubit3: QubitSpecifier, target_qubit: QubitSpecifier) -> InstructionSet:
    """Apply :class:`~qiskit.circuit.library.RC3XGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            control_qubit3: The qubit(s) used as the third control.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
    from .library.standard_gates.x import RC3XGate
    return self.append(RC3XGate(), [control_qubit1, control_qubit2, control_qubit3, target_qubit], [])