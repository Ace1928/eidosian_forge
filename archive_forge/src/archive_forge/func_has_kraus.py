from typing import Any, Sequence, Tuple, TypeVar, Union
import warnings
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.mixture_protocol import has_mixture
from cirq.type_workarounds import NotImplementedType
def has_kraus(val: Any, *, allow_decompose: bool=True) -> bool:
    """Returns whether the value has a Kraus representation.

    Args:
        val: The value to check.
        allow_decompose: Used by internal methods to stop redundant
            decompositions from being performed (e.g. there's no need to
            decompose an object to check if it is unitary as part of determining
            if the object is a quantum channel, when the quantum channel check
            will already be doing a more general decomposition check). Defaults
            to True. When False, the decomposition strategy for determining
            the result is skipped.

    Returns:
        If `val` has a `_has_kraus_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if `val` has a
        `_has_mixture_` method and its result is not NotImplemented, that
        result is returned. Otherwise if `val` has a `_has_unitary_` method
        and its results is not NotImplemented, that result is returned.
        Otherwise, if the value has a _kraus_ method return if that
        has a non-default value. Returns False if none of these functions
        exists.
    """
    kraus_getter = getattr(val, '_has_kraus_', None)
    result = NotImplemented if kraus_getter is None else kraus_getter()
    if result is not NotImplemented:
        return result
    result = has_mixture(val, allow_decompose=False)
    if result is not NotImplemented and result:
        return result
    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all((has_kraus(val) for val in operations))
    return kraus(val, None) is not None