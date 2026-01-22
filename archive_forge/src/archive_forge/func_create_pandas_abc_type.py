from __future__ import annotations
from typing import (
def create_pandas_abc_type(name, attr, comp):

    def _check(inst) -> bool:
        return getattr(inst, attr, '_typ') in comp

    @classmethod
    def _instancecheck(cls, inst) -> bool:
        return _check(inst) and (not isinstance(inst, type))

    @classmethod
    def _subclasscheck(cls, inst) -> bool:
        if not isinstance(inst, type):
            raise TypeError('issubclass() arg 1 must be a class')
        return _check(inst)
    dct = {'__instancecheck__': _instancecheck, '__subclasscheck__': _subclasscheck}
    meta = type('ABCBase', (type,), dct)
    return meta(name, (), dct)