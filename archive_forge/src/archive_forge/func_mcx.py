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
def mcx(self, control_qubits: Sequence[QubitSpecifier], target_qubit: QubitSpecifier, ancilla_qubits: QubitSpecifier | Sequence[QubitSpecifier] | None=None, mode: str='noancilla') -> InstructionSet:
    """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:

        - ``'noancilla'``: Requires 0 ancilla qubits.
        - ``'recursion'``: Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - ``'v-chain'``: Requires 2 less ancillas than the number of control qubits.
        - ``'v-chain-dirty'``: Same as for the clean ancillas (but the circuit will be longer).

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            ancilla_qubits: The qubits used as the ancillae, if the mode requires them.
            mode: The choice of mode, explained further above.

        Returns:
            A handle to the instructions created.

        Raises:
            ValueError: if the given mode is not known, or if too few ancilla qubits are passed.
            AttributeError: if no ancilla qubits are passed, but some are needed.
        """
    from .library.standard_gates.x import MCXGrayCode, MCXRecursive, MCXVChain
    num_ctrl_qubits = len(control_qubits)
    available_implementations = {'noancilla': MCXGrayCode(num_ctrl_qubits), 'recursion': MCXRecursive(num_ctrl_qubits), 'v-chain': MCXVChain(num_ctrl_qubits, False), 'v-chain-dirty': MCXVChain(num_ctrl_qubits, dirty_ancillas=True), 'advanced': MCXRecursive(num_ctrl_qubits), 'basic': MCXVChain(num_ctrl_qubits, dirty_ancillas=False), 'basic-dirty-ancilla': MCXVChain(num_ctrl_qubits, dirty_ancillas=True)}
    if ancilla_qubits:
        _ = self.qbit_argument_conversion(ancilla_qubits)
    try:
        gate = available_implementations[mode]
    except KeyError as ex:
        all_modes = list(available_implementations.keys())
        raise ValueError(f'Unsupported mode ({mode}) selected, choose one of {all_modes}') from ex
    if hasattr(gate, 'num_ancilla_qubits') and gate.num_ancilla_qubits > 0:
        required = gate.num_ancilla_qubits
        if ancilla_qubits is None:
            raise AttributeError(f'No ancillas provided, but {required} are needed!')
        if not hasattr(ancilla_qubits, '__len__'):
            ancilla_qubits = [ancilla_qubits]
        if len(ancilla_qubits) < required:
            actually = len(ancilla_qubits)
            raise ValueError(f'At least {required} ancillas required, but {actually} given.')
        ancilla_qubits = ancilla_qubits[:required]
    else:
        ancilla_qubits = []
    return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])