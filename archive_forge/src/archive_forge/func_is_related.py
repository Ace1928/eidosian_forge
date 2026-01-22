from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def is_related(typ):
    return typ not in bases and hasattr(typ, '__mro__') and (not isinstance(typ, GenericAlias)) and issubclass(cls, typ)