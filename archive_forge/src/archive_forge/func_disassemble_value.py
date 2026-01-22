import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
@staticmethod
def disassemble_value(value_expr: Union[float, str]) -> Union[float, ParameterExpression]:
    """A helper function to format instruction operand.

        If parameter in string representation is specified, this method parses the
        input string and generates Qiskit ParameterExpression object.

        Args:
            value_expr: Operand value in Qobj.

        Returns:
            Parsed operand value. ParameterExpression object is returned if value is not number.
        """
    if isinstance(value_expr, str):
        str_expr = parse_string_expr(value_expr, partial_binding=False)
        value_expr = str_expr(**{pname: Parameter(pname) for pname in str_expr.params})
    return value_expr