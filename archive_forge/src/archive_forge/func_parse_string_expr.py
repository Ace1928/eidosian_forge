from __future__ import annotations
import ast
import copy
import operator
import cmath
from collections.abc import Callable
from typing import Any
from qiskit.pulse.exceptions import PulseError
from qiskit.circuit import ParameterExpression
def parse_string_expr(source: str, partial_binding: bool=False) -> PulseExpression:
    """Safe parsing of string expression.

    Args:
        source: String expression to parse.
        partial_binding: Allow partial bind of parameters.

    Returns:
        PulseExpression: Returns a expression object.

    Example:

        expr = 'P1 + P2 + P3'
        parsed_expr = parse_string_expr(expr, partial_binding=True)

        # create new PulseExpression
        bound_two = parsed_expr(P1=1, P2=2)
        # evaluate expression
        value1 = bound_two(P3=3)
        value2 = bound_two(P3=4)
        value3 = bound_two(P3=5)

    """
    subs = [('numpy.', ''), ('np.', ''), ('math.', ''), ('cmath.', '')]
    for match, sub in subs:
        source = source.replace(match, sub)
    return PulseExpression(source, partial_binding)