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
def repr_excinfo(self, excinfo: ExceptionInfo[BaseException]) -> 'ExceptionChainRepr':
    repr_chain: List[Tuple[ReprTraceback, Optional[ReprFileLocation], Optional[str]]] = []
    e: Optional[BaseException] = excinfo.value
    excinfo_: Optional[ExceptionInfo[BaseException]] = excinfo
    descr = None
    seen: Set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if excinfo_:
            if isinstance(e, BaseExceptionGroup):
                reprtraceback: Union[ReprTracebackNative, ReprTraceback] = ReprTracebackNative(traceback.format_exception(type(excinfo_.value), excinfo_.value, excinfo_.traceback[0]._rawentry))
            else:
                reprtraceback = self.repr_traceback(excinfo_)
            reprcrash = excinfo_._getreprcrash()
        else:
            reprtraceback = ReprTracebackNative(traceback.format_exception(type(e), e, None))
            reprcrash = None
        repr_chain += [(reprtraceback, reprcrash, descr)]
        if e.__cause__ is not None and self.chain:
            e = e.__cause__
            excinfo_ = ExceptionInfo.from_exception(e) if e.__traceback__ else None
            descr = 'The above exception was the direct cause of the following exception:'
        elif e.__context__ is not None and (not e.__suppress_context__) and self.chain:
            e = e.__context__
            excinfo_ = ExceptionInfo.from_exception(e) if e.__traceback__ else None
            descr = 'During handling of the above exception, another exception occurred:'
        else:
            e = None
    repr_chain.reverse()
    return ExceptionChainRepr(repr_chain)