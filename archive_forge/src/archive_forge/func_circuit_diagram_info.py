import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def circuit_diagram_info(val: Any, args: Optional[CircuitDiagramInfoArgs]=None, default=RaiseTypeErrorIfNotProvided):
    """Requests information on drawing an operation in a circuit diagram.

    Calls _circuit_diagram_info_ on `val`. If `val` doesn't have
    _circuit_diagram_info_, or it returns NotImplemented, that indicates that
    diagram information is not available.

    Args:
        val: The operation or gate that will need to be drawn.
        args: A CircuitDiagramInfoArgs describing the desired drawing style.
        default: A default result to return if the value doesn't have circuit
            diagram information. If not specified, a TypeError is raised
            instead.

    Returns:
        If `val` has no _circuit_diagram_info_ method or it returns
        NotImplemented, then `default` is returned (or a TypeError is
        raised if no `default` is specified).

        Otherwise, the value returned by _circuit_diagram_info_ is returned.

    Raises:
        TypeError:
            `val` doesn't have circuit diagram information and `default` was
            not specified.
    """
    if args is None:
        args = CircuitDiagramInfoArgs.UNINFORMED_DEFAULT
    getter = getattr(val, '_circuit_diagram_info_', None)
    result = NotImplemented if getter is None else getter(args)
    if isinstance(result, str):
        return CircuitDiagramInfo(wire_symbols=(result,))
    if isinstance(result, Iterable):
        return CircuitDiagramInfo(wire_symbols=tuple(result))
    if result is not NotImplemented:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if getter is None:
        raise TypeError(f"object of type '{type(val)}' has no _circuit_diagram_info_ method.")
    raise TypeError(f"object of type '{type(val)}' does have a _circuit_diagram_info_ method, but it returned NotImplemented.")