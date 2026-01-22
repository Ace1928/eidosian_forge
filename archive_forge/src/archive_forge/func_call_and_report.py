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
def call_and_report(item: Item, when: Literal['setup', 'call', 'teardown'], log: bool=True, **kwds) -> TestReport:
    ihook = item.ihook
    if when == 'setup':
        runtest_hook: Callable[..., None] = ihook.pytest_runtest_setup
    elif when == 'call':
        runtest_hook = ihook.pytest_runtest_call
    elif when == 'teardown':
        runtest_hook = ihook.pytest_runtest_teardown
    else:
        assert False, f'Unhandled runtest hook case: {when}'
    reraise: Tuple[Type[BaseException], ...] = (Exit,)
    if not item.config.getoption('usepdb', False):
        reraise += (KeyboardInterrupt,)
    call = CallInfo.from_call(lambda: runtest_hook(item=item, **kwds), when=when, reraise=reraise)
    report: TestReport = ihook.pytest_runtest_makereport(item=item, call=call)
    if log:
        ihook.pytest_runtest_logreport(report=report)
    if check_interactive_exception(call, report):
        ihook.pytest_exception_interact(node=item, call=call, report=report)
    return report