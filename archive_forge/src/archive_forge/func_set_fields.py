from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def set_fields(dct, bases, name):
    dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
    for k, v in list(dct.items()):
        if isinstance(v, _PField):
            dct[name][k] = v
            del dct[k]