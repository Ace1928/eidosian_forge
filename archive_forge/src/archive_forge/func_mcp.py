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
def mcp(self, lam: ParameterValueType, control_qubits: Sequence[QubitSpecifier], target_qubit: QubitSpecifier) -> InstructionSet:
    """Apply :class:`~qiskit.circuit.library.MCPhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            lam: The angle of the rotation.
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
    from .library.standard_gates.p import MCPhaseGate
    num_ctrl_qubits = len(control_qubits)
    return self.append(MCPhaseGate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])