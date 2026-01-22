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
def get_return_annotation(self) -> Optional[AccessPath]:
    try:
        o = self._obj.__annotations__.get('return')
    except AttributeError:
        return None
    if o is None:
        return None
    try:
        o = typing.get_type_hints(self._obj).get('return')
    except Exception:
        pass
    return self._create_access_path(o)