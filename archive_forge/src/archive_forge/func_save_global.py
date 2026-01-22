import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def save_global(self, obj, name=None, pack=struct.pack):
    """
            Save a "global".

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
    if obj is type(None):
        return self.save_reduce(type, (None,), obj=obj)
    elif obj is type(Ellipsis):
        return self.save_reduce(type, (Ellipsis,), obj=obj)
    elif obj is type(NotImplemented):
        return self.save_reduce(type, (NotImplemented,), obj=obj)
    elif obj in _BUILTIN_TYPE_NAMES:
        return self.save_reduce(_builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)
    if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):
        self.save_reduce(_create_parametrized_type_hint, parametrized_type_hint_getinitargs(obj), obj=obj)
    elif name is not None:
        Pickler.save_global(self, obj, name=name)
    elif not _should_pickle_by_reference(obj, name=name):
        self._save_reduce_pickle5(*_dynamic_class_reduce(obj), obj=obj)
    else:
        Pickler.save_global(self, obj, name=name)