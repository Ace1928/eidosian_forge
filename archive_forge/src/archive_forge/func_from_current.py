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
@classmethod
def from_current(cls, exprinfo: Optional[str]=None) -> 'ExceptionInfo[BaseException]':
    """Return an ExceptionInfo matching the current traceback.

        .. warning::

            Experimental API

        :param exprinfo:
            A text string helping to determine if we should strip
            ``AssertionError`` from the output. Defaults to the exception
            message/``__str__()``.
        """
    tup = sys.exc_info()
    assert tup[0] is not None, 'no current exception'
    assert tup[1] is not None, 'no current exception'
    assert tup[2] is not None, 'no current exception'
    exc_info = (tup[0], tup[1], tup[2])
    return ExceptionInfo.from_exc_info(exc_info, exprinfo)