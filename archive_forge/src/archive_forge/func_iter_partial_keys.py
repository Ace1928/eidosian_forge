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
def iter_partial_keys():
    for i, k in enumerate(self._obj.keys()):
        if i > 50:
            break
        yield k