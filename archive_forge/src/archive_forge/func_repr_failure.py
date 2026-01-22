import bdb
import builtins
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex, import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
def repr_failure(self, excinfo: ExceptionInfo[BaseException]) -> Union[str, TerminalRepr]:
    import doctest
    failures: Optional[Sequence[Union[doctest.DocTestFailure, doctest.UnexpectedException]]] = None
    if isinstance(excinfo.value, (doctest.DocTestFailure, doctest.UnexpectedException)):
        failures = [excinfo.value]
    elif isinstance(excinfo.value, MultipleDoctestFailures):
        failures = excinfo.value.failures
    if failures is None:
        return super().repr_failure(excinfo)
    reprlocation_lines = []
    for failure in failures:
        example = failure.example
        test = failure.test
        filename = test.filename
        if test.lineno is None:
            lineno = None
        else:
            lineno = test.lineno + example.lineno + 1
        message = type(failure).__name__
        reprlocation = ReprFileLocation(filename, lineno, message)
        checker = _get_checker()
        report_choice = _get_report_choice(self.config.getoption('ipdoctestreport'))
        if lineno is not None:
            assert failure.test.docstring is not None
            lines = failure.test.docstring.splitlines(False)
            assert test.lineno is not None
            lines = ['%03d %s' % (i + test.lineno + 1, x) for i, x in enumerate(lines)]
            lines = lines[max(example.lineno - 9, 0):example.lineno + 1]
        else:
            lines = ['EXAMPLE LOCATION UNKNOWN, not showing all tests of that example']
            indent = '>>>'
            for line in example.source.splitlines():
                lines.append(f'??? {indent} {line}')
                indent = '...'
        if isinstance(failure, doctest.DocTestFailure):
            lines += checker.output_difference(example, failure.got, report_choice).split('\n')
        else:
            inner_excinfo = ExceptionInfo.from_exc_info(failure.exc_info)
            lines += ['UNEXPECTED EXCEPTION: %s' % repr(inner_excinfo.value)]
            lines += [x.strip('\n') for x in traceback.format_exception(*failure.exc_info)]
        reprlocation_lines.append((reprlocation, lines))
    return ReprFailDoctest(reprlocation_lines)