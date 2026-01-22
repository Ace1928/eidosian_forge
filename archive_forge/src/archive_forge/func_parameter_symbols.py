import numbers
from typing import AbstractSet, Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self
import sympy
from typing_extensions import Protocol
from cirq import study
from cirq._doc import doc_private
def parameter_symbols(val: Any) -> AbstractSet[sympy.Symbol]:
    """Returns parameter symbols for this object.

    Args:
        val: Object for which to find the parameter symbols.

    Returns:
        A set of parameter symbols if the object is parameterized. It the object
        does not implement the _parameter_symbols_ magic method or that method
        returns NotImplemented, returns an empty set.
    """
    return {sympy.Symbol(name) for name in parameter_names(val)}