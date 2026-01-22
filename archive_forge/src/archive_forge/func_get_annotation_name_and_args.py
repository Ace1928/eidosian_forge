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
def get_annotation_name_and_args(self):
    """
        Returns Tuple[Optional[str], Tuple[AccessPath, ...]]
        """
    name = None
    args = ()
    if safe_getattr(self._obj, '__module__', default='') == 'typing':
        m = re.match('typing.(\\w+)\\[', repr(self._obj))
        if m is not None:
            name = m.group(1)
            import typing
            if sys.version_info >= (3, 8):
                args = typing.get_args(self._obj)
            else:
                args = safe_getattr(self._obj, '__args__', default=None)
    return (name, tuple((self._create_access_path(arg) for arg in args)))