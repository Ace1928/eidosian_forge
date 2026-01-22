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
def get_wire_map(circuit, bits, cregbundle):
    """Map the bits and registers to the index from the top of the drawing.
    The key to the dict is either the (Qubit, Clbit) or if cregbundle True,
    the register that is being bundled.

    Args:
        circuit (QuantumCircuit): the circuit being drawn
        bits (list(Qubit, Clbit)): the Qubit's and Clbit's in the circuit
        cregbundle (bool): if True bundle classical registers. Default: ``True``.

    Returns:
        dict((Qubit, Clbit, ClassicalRegister): index): map of bits/registers
            to index
    """
    prev_reg = None
    wire_index = 0
    wire_map = {}
    for bit in bits:
        register = get_bit_register(circuit, bit)
        if register is None or not isinstance(bit, Clbit) or (not cregbundle):
            wire_map[bit] = wire_index
            wire_index += 1
        elif register is not None and cregbundle and (register != prev_reg):
            prev_reg = register
            wire_map[register] = wire_index
            wire_index += 1
    return wire_map