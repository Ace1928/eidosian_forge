import bdb
import dataclasses
import os
import sys
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import Generic
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .reports import BaseReport
from .reports import CollectErrorRepr
from .reports import CollectReport
from .reports import TestReport
from _pytest import timing
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.nodes import Collector
from _pytest.nodes import Directory
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.outcomes import Exit
from _pytest.outcomes import OutcomeException
from _pytest.outcomes import Skipped
from _pytest.outcomes import TEST_OUTCOME
def check_interactive_exception(call: 'CallInfo[object]', report: BaseReport) -> bool:
    """Check whether the call raised an exception that should be reported as
    interactive."""
    if call.excinfo is None:
        return False
    if hasattr(report, 'wasxfail'):
        return False
    if isinstance(call.excinfo.value, (Skipped, bdb.BdbQuit)):
        return False
    return True