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
@classmethod
def from_item_and_call(cls, item: Item, call: 'CallInfo[None]') -> 'TestReport':
    """Create and fill a TestReport with standard item and call info.

        :param item: The item.
        :param call: The call info.
        """
    when = call.when
    assert when != 'collect'
    duration = call.duration
    start = call.start
    stop = call.stop
    keywords = {x: 1 for x in item.keywords}
    excinfo = call.excinfo
    sections = []
    if not call.excinfo:
        outcome: Literal['passed', 'failed', 'skipped'] = 'passed'
        longrepr: Union[None, ExceptionInfo[BaseException], Tuple[str, int, str], str, TerminalRepr] = None
    elif not isinstance(excinfo, ExceptionInfo):
        outcome = 'failed'
        longrepr = excinfo
    elif isinstance(excinfo.value, skip.Exception):
        outcome = 'skipped'
        r = excinfo._getreprcrash()
        assert r is not None, 'There should always be a traceback entry for skipping a test.'
        if excinfo.value._use_item_location:
            path, line = item.reportinfo()[:2]
            assert line is not None
            longrepr = (os.fspath(path), line + 1, r.message)
        else:
            longrepr = (str(r.path), r.lineno, r.message)
    else:
        outcome = 'failed'
        if call.when == 'call':
            longrepr = item.repr_failure(excinfo)
        else:
            longrepr = item._repr_failure_py(excinfo, style=item.config.getoption('tbstyle', 'auto'))
    for rwhen, key, content in item._report_sections:
        sections.append((f'Captured {key} {rwhen}', content))
    return cls(item.nodeid, item.location, keywords, outcome, longrepr, when, sections, duration, start, stop, user_properties=item.user_properties)