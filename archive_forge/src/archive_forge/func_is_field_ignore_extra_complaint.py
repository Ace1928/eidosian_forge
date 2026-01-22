from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def is_field_ignore_extra_complaint(type_cls, field, ignore_extra):
    if not ignore_extra:
        return False
    if not is_type_cls(type_cls, field.type):
        return False
    return 'ignore_extra' in inspect.signature(field.factory).parameters