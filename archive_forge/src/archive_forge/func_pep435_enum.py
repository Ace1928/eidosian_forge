from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
def pep435_enum(name):
    __members__ = OrderedDict()

    def __init__(self, name, value, alias=None):
        self.name = name
        self.value = value
        self.__members__[name] = self
        value_to_member[value] = self
        setattr(self.__class__, name, self)
        if alias:
            self.__members__[alias] = self
            setattr(self.__class__, alias, self)
    value_to_member = {}

    @classmethod
    def get(cls, value):
        return value_to_member[value]
    someenum = type(name, (object,), {'__members__': __members__, '__init__': __init__, 'get': get})
    try:
        module = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    if module is not None:
        someenum.__module__ = module
    return someenum