from __future__ import annotations
from typing import TYPE_CHECKING
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.exceptions import CircuitError
from ._builder_utils import validate_condition, condition_resources
from .control_flow import ControlFlowOp
def c_if(self, classical, val):
    raise NotImplementedError('WhileLoopOp cannot be classically controlled through Instruction.c_if. Please use an IfElseOp instead.')