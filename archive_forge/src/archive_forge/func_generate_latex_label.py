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
@_optionals.HAS_PYLATEX.require_in_call('the latex and latex_source circuit drawers')
def generate_latex_label(label):
    """Convert a label to a valid latex string."""
    from pylatexenc.latexencode import utf8tolatex
    regex = re.compile('(?<!\\\\)\\$(.*)(?<!\\\\)\\$')
    match = regex.search(label)
    if not match:
        label = label.replace('\\$', '$')
        final_str = utf8tolatex(label, non_ascii_only=True)
    else:
        mathmode_string = match.group(1).replace('\\$', '$')
        before_match = label[:match.start()]
        before_match = before_match.replace('\\$', '$')
        after_match = label[match.end():]
        after_match = after_match.replace('\\$', '$')
        final_str = utf8tolatex(before_match, non_ascii_only=True) + mathmode_string + utf8tolatex(after_match, non_ascii_only=True)
    return final_str.replace(' ', '\\,')