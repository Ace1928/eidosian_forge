import dataclasses
from io import StringIO
import os
from pprint import pprint
from typing import Any
from typing import cast
from typing import Dict
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprEntry
from _pytest._code.code import ReprEntryNative
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import ReprFuncArgs
from _pytest._code.code import ReprLocals
from _pytest._code.code import ReprTraceback
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import skip
def deserialize_repr_entry(entry_data):
    data = entry_data['data']
    entry_type = entry_data['type']
    if entry_type == 'ReprEntry':
        reprfuncargs = None
        reprfileloc = None
        reprlocals = None
        if data['reprfuncargs']:
            reprfuncargs = ReprFuncArgs(**data['reprfuncargs'])
        if data['reprfileloc']:
            reprfileloc = ReprFileLocation(**data['reprfileloc'])
        if data['reprlocals']:
            reprlocals = ReprLocals(data['reprlocals']['lines'])
        reprentry: Union[ReprEntry, ReprEntryNative] = ReprEntry(lines=data['lines'], reprfuncargs=reprfuncargs, reprlocals=reprlocals, reprfileloc=reprfileloc, style=data['style'])
    elif entry_type == 'ReprEntryNative':
        reprentry = ReprEntryNative(data['lines'])
    else:
        _report_unserialization_failure(entry_type, TestReport, reportdict)
    return reprentry