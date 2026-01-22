from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def inject_param_text(doctext: str, inject_params: Dict[str, str]) -> str:
    doclines = collections.deque(doctext.splitlines())
    lines = []
    to_inject = None
    while doclines:
        line = doclines.popleft()
        m = _param_reg.match(line)
        if to_inject is None:
            if m:
                param = m.group(2).lstrip('*')
                if param in inject_params:
                    indent = ' ' * len(m.group(1)) + ' '
                    if doclines:
                        m2 = re.match('(\\s+)\\S', doclines[0])
                        if m2:
                            indent = ' ' * len(m2.group(1))
                    to_inject = indent + inject_params[param]
        elif m:
            lines.extend(['\n', to_inject, '\n'])
            to_inject = None
        elif not line.rstrip():
            lines.extend([line, to_inject, '\n'])
            to_inject = None
        elif line.endswith('::'):
            lines.extend([line, doclines.popleft()])
            continue
        lines.append(line)
    return '\n'.join(lines)