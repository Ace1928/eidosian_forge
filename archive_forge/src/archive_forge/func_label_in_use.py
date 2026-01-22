from __future__ import annotations
import contextlib
from typing import Union, Iterable, Any, Tuple, Optional, List, Literal, TYPE_CHECKING
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources
def label_in_use(self, label):
    """Return whether a case label is already accounted for in the switch statement."""
    return label in self._label_set