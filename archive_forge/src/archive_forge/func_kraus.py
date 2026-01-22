from typing import Any, Sequence, Tuple, TypeVar, Union
import warnings
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.mixture_protocol import has_mixture
from cirq.type_workarounds import NotImplementedType
def kraus(val: Any, default: Any=RaiseTypeErrorIfNotProvided) -> Union[Tuple[np.ndarray, ...], TDefault]:
    """Returns a list of matrices describing the channel for the given value.

    These matrices are the terms in the operator sum representation of
    a quantum channel. If the returned matrices are ${A_0,A_1,..., A_{r-1}}$,
    then this describes the channel:
        $$
        \\rho \\rightarrow \\sum_{k=0}^{r-1} A_k \\rho A_k^\\dagger
        $$
    These matrices are required to satisfy the trace preserving condition
        $$
        \\sum_{k=0}^{r-1} A_k^\\dagger A_k = I
        $$
    where $I$ is the identity matrix. The matrices $A_k$ are sometimes called
    Kraus or noise operators.

    Args:
        val: The value to describe by a channel.
        default: Determines the fallback behavior when `val` doesn't have
            a channel. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.

    Returns:
        If `val` has a `_kraus_` method and its result is not NotImplemented,
        that result is returned. Otherwise, if `val` has a `_mixture_` method
        and its results is not NotImplement a tuple made up of channel
        corresponding to that mixture being a probabilistic mixture of unitaries
        is returned.  Otherwise, if `val` has a `_unitary_` method and
        its result is not NotImplemented a tuple made up of that result is
        returned. Otherwise, if a default value was specified, the default
        value is returned.

    Raises:
        TypeError: `val` doesn't have a _kraus_ or _unitary_ method (or that
            method returned NotImplemented) and also no default value was
            specified.
    """
    channel_getter = getattr(val, '_channel_', None)
    if channel_getter is not None:
        warnings.warn('_channel_ is deprecated and will be removed in cirq 0.13, rename to _kraus_', DeprecationWarning)
    kraus_getter = getattr(val, '_kraus_', None)
    kraus_result = NotImplemented if kraus_getter is None else kraus_getter()
    if kraus_result is not NotImplemented:
        return tuple(kraus_result)
    mixture_getter = getattr(val, '_mixture_', None)
    mixture_result = NotImplemented if mixture_getter is None else mixture_getter()
    if mixture_result is not NotImplemented and mixture_result is not None:
        return tuple((np.sqrt(p) * u for p, u in mixture_result))
    unitary_getter = getattr(val, '_unitary_', None)
    unitary_result = NotImplemented if unitary_getter is None else unitary_getter()
    if unitary_result is not NotImplemented and unitary_result is not None:
        return (unitary_result,)
    channel_result = NotImplemented if channel_getter is None else channel_getter()
    if channel_result is not NotImplemented:
        return tuple(channel_result)
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if kraus_getter is None and unitary_getter is None and (mixture_getter is None):
        raise TypeError(f"object of type '{type(val)}' has no _kraus_ or _mixture_ or _unitary_ method.")
    raise TypeError(f"object of type '{type(val)}' does have a _kraus_, _mixture_ or _unitary_ method, but it returned NotImplemented.")