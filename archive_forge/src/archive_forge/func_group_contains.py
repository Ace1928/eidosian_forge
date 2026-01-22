import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
def group_contains(self, expected_exception: Union[Type[BaseException], Tuple[Type[BaseException], ...]], *, match: Union[str, Pattern[str], None]=None, depth: Optional[int]=None) -> bool:
    """Check whether a captured exception group contains a matching exception.

        :param Type[BaseException] | Tuple[Type[BaseException]] expected_exception:
            The expected exception type, or a tuple if one of multiple possible
            exception types are expected.

        :param str | Pattern[str] | None match:
            If specified, a string containing a regular expression,
            or a regular expression object, that is tested against the string
            representation of the exception and its `PEP-678 <https://peps.python.org/pep-0678/>` `__notes__`
            using :func:`re.search`.

            To match a literal string that may contain :ref:`special characters
            <re-syntax>`, the pattern can first be escaped with :func:`re.escape`.

        :param Optional[int] depth:
            If `None`, will search for a matching exception at any nesting depth.
            If >= 1, will only match an exception if it's at the specified depth (depth = 1 being
            the exceptions contained within the topmost exception group).
        """
    msg = 'Captured exception is not an instance of `BaseExceptionGroup`'
    assert isinstance(self.value, BaseExceptionGroup), msg
    msg = '`depth` must be >= 1 if specified'
    assert depth is None or depth >= 1, msg
    return self._group_contains(self.value, expected_exception, match, depth)