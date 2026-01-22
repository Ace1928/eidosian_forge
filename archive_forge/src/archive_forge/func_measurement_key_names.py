from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def measurement_key_names(val: Any) -> FrozenSet[str]:
    """Gets the measurement key strings of measurements within the given value.

    Args:
        val: The value which has the measurement key.
        allow_decompose: Defaults to True. When true, composite operations that
            don't directly specify their measurement keys will be decomposed in
            order to find measurement keys within the decomposed operations. If
            not set, composite operations will appear to have no measurement
            keys. Used by internal methods to stop redundant decompositions from
            being performed.

    Returns:
        The measurement keys of the value. If the value has no measurement,
        the result is the empty set.
    """
    result = _measurement_key_names_from_magic_methods(val)
    if result is not NotImplemented and result is not None:
        return result
    key_objs = _measurement_key_objs_from_magic_methods(val)
    if key_objs is not NotImplemented and key_objs is not None:
        return frozenset((str(key_obj) for key_obj in key_objs))
    return frozenset()