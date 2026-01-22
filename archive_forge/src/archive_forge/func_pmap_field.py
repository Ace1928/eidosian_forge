from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def pmap_field(key_type, value_type, optional=False, invariant=PFIELD_NO_INVARIANT):
    """
    Create a checked ``PMap`` field.

    :param key: The required type for the keys of the map.
    :param value: The required type for the values of the map.
    :param optional: If true, ``None`` can be used as a value for
        this field.
    :param invariant: Pass-through to ``field``.

    :return: A ``field`` containing a ``CheckedPMap``.
    """
    TheMap = _make_pmap_field_type(key_type, value_type)
    if optional:

        def factory(argument):
            if argument is None:
                return None
            else:
                return TheMap.create(argument)
    else:
        factory = TheMap.create
    return field(mandatory=True, initial=TheMap(), type=optional_type(TheMap) if optional else TheMap, factory=factory, invariant=invariant)