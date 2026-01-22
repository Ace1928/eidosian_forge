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
def remove_final_measurements(self, inplace: bool=True) -> Optional['QuantumCircuit']:
    """Removes final measurements and barriers on all qubits if they are present.
        Deletes the classical registers that were used to store the values from these measurements
        that become idle as a result of this operation, and deletes classical bits that are
        referenced only by removed registers, or that aren't referenced at all but have
        become idle as a result of this operation.

        Measurements and barriers are considered final if they are
        followed by no other operations (aside from other measurements or barriers.)

        Args:
            inplace (bool): All measurements removed inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns the resulting circuit when ``inplace=False``, else None.
        """
    from qiskit.transpiler.passes import RemoveFinalMeasurements
    from qiskit.converters import circuit_to_dag
    if inplace:
        circ = self
    else:
        circ = self.copy()
    dag = circuit_to_dag(circ)
    remove_final_meas = RemoveFinalMeasurements()
    new_dag = remove_final_meas.run(dag)
    kept_cregs = set(new_dag.cregs.values())
    kept_clbits = set(new_dag.clbits)
    cregs_to_add = [creg for creg in circ.cregs if creg in kept_cregs]
    clbits_to_add = [clbit for clbit in circ._data.clbits if clbit in kept_clbits]
    circ.cregs = []
    circ._clbit_indices = {}
    circ._data = CircuitData(qubits=circ._data.qubits, reserve=len(circ._data))
    circ._parameter_table.clear()
    circ.global_phase = circ.global_phase
    circ.add_bits(clbits_to_add)
    for creg in cregs_to_add:
        circ.add_register(creg)
    for node in new_dag.topological_op_nodes():
        inst = node.op.copy()
        circ.append(inst, node.qargs, node.cargs)
    if not inplace:
        return circ
    else:
        return None