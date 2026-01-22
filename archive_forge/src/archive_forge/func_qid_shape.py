from typing import Any, Sequence, Tuple, TypeVar, Union
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import document, doc_private
from cirq.type_workarounds import NotImplementedType
def qid_shape(val: Any, default: TDefault=RaiseTypeErrorIfNotProvided) -> Union[Tuple[int, ...], TDefault]:
    """Returns a tuple describing the number of quantum levels of each
    qubit/qudit/qid `val` operates on.

    Args:
        val: The value to get the shape of.
        default: Determines the fallback behavior when `val` doesn't have
            a shape. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.

    Returns:
        If `val` has a `_qid_shape_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if `val` has a
        `_num_qubits_` method, the shape with `num_qubits` qubits is returned
        e.g. `(2,)*num_qubits`. If neither method returns a value other than
        NotImplemented and a default value was specified, the default value is
        returned.

    Raises:
        TypeError: `val` doesn't have either a `_qid_shape_` or a `_num_qubits_`
            method (or they returned NotImplemented) and also no default value
            was specified.
    """
    getter = getattr(val, '_qid_shape_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented:
        return result
    if isinstance(val, Sequence) and all((isinstance(q, ops.Qid) for q in val)):
        return tuple((q.dimension for q in val))
    num_getter = getattr(val, '_num_qubits_', None)
    num_qubits = NotImplemented if num_getter is None else num_getter()
    if num_qubits is not NotImplemented:
        return (2,) * num_qubits
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if getter is not None:
        raise TypeError(f"object of type '{type(val)}' does have a _qid_shape_ method, but it returned NotImplemented.")
    if num_getter is not None:
        raise TypeError(f"object of type '{type(val)}' does have a _num_qubits_ method, but it returned NotImplemented.")
    raise TypeError(f"object of type '{type(val)}' has no _num_qubits_ or _qid_shape_ methods.")