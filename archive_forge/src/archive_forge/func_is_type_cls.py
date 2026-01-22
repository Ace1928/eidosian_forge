from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def is_type_cls(type_cls, field_type):
    if type(field_type) is set:
        return True
    types = tuple(field_type)
    if len(types) == 0:
        return False
    return issubclass(get_type(types[0]), type_cls)