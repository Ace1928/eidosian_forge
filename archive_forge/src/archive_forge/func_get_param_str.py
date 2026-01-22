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
def get_param_str(op, drawer, ndigits=3):
    """Get the params as a string to add to the gate text display"""
    if not hasattr(op, 'params') or any((isinstance(param, np.ndarray) for param in op.params)) or any((isinstance(param, QuantumCircuit) for param in op.params)):
        return ''
    if isinstance(op, Delay):
        param_list = [f'{op.params[0]}[{op.unit}]']
    else:
        param_list = []
        for count, param in enumerate(op.params):
            if drawer == 'latex' and count > 3 or (drawer == 'mpl' and count > 15):
                param_list.append('...')
                break
            try:
                param_list.append(pi_check(param, output=drawer, ndigits=ndigits))
            except TypeError:
                param_list.append(str(param))
    param_str = ''
    if param_list:
        if drawer == 'latex':
            param_str = f'\\,(\\mathrm{{{','.join(param_list)}}})'
        elif drawer == 'mpl':
            param_str = f'{', '.join(param_list)}'.replace('-', '$-$')
        else:
            param_str = f'({','.join(param_list)})'
    return param_str