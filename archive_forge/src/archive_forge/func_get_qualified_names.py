import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def get_qualified_names(self):

    def try_to_get_name(obj):
        return getattr(obj, '__qualname__', getattr(obj, '__name__', None))
    if self.is_module():
        return ()
    name = try_to_get_name(self._obj)
    if name is None:
        name = try_to_get_name(type(self._obj))
        if name is None:
            return ()
    return tuple(name.split('.'))