from the parameter table model of ~O(1), however, usually, this calculation occurs
from each object, yielding smaller object creation cost and higher performance
from __future__ import annotations
from copy import copy
from typing import Any
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import instructions, channels
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import SymbolicPulse, Waveform
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
from qiskit.pulse.utils import format_parameter_value
def visit_ScheduleBlock(self, node: ScheduleBlock):
    """Visit ``ScheduleBlock``. Recursively visit context blocks and search parameters.

        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.
        """
    self.parameters |= node._parameter_manager.parameters