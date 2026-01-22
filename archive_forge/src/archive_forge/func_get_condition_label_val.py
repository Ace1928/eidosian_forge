import re
from collections import OrderedDict
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit, ControlFlowOp
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, PowerModifier
from qiskit.circuit.tools import pi_check
from qiskit.converters import circuit_to_dag
from qiskit.utils import optionals as _optionals
from ..exceptions import VisualizationError
def get_condition_label_val(condition, circuit, cregbundle):
    """Get the label and value list to display a condition

    Args:
        condition (Union[Clbit, ClassicalRegister], int): classical condition
        circuit (QuantumCircuit): the circuit that is being drawn
        cregbundle (bool): if set True bundle classical registers

    Returns:
        str: label to display for the condition
        list(str): list of 1's and 0's indicating values of condition
    """
    cond_is_bit = bool(isinstance(condition[0], Clbit))
    cond_val = int(condition[1])
    if isinstance(condition[0], ClassicalRegister) and (not cregbundle):
        val_bits = list(f'{cond_val:0{condition[0].size}b}')[::-1]
    else:
        val_bits = list(str(cond_val))
    label = ''
    if cond_is_bit and cregbundle:
        register, _, reg_index = get_bit_reg_index(circuit, condition[0])
        if register is not None:
            label = f'{register.name}_{reg_index}={hex(cond_val)}'
    elif not cond_is_bit:
        label = hex(cond_val)
    return (label, val_bits)