import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def pickle_by_enum_name(self, proto):
    return (getattr, (self.__class__, self._name_))