import argparse
from collections import Counter
import dataclasses
import datetime
from functools import partial
import inspect
from pathlib import Path
import platform
import sys
import textwrap
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import final
from typing import Generator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from _pytest import nodes
from _pytest import timing
from _pytest._code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._io import TerminalWriter
from _pytest._io.wcwidth import wcswidth
import _pytest._version
from _pytest.assertion.util import running_on_ci
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.reports import BaseReport
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
def report_collect(self, final: bool=False) -> None:
    if self.config.option.verbose < 0:
        return
    if not final:
        t = timing.time()
        if self._collect_report_last_write is not None and self._collect_report_last_write > t - REPORT_COLLECTING_RESOLUTION:
            return
        self._collect_report_last_write = t
    errors = len(self.stats.get('error', []))
    skipped = len(self.stats.get('skipped', []))
    deselected = len(self.stats.get('deselected', []))
    selected = self._numcollected - deselected
    line = 'collected ' if final else 'collecting '
    line += str(self._numcollected) + ' item' + ('' if self._numcollected == 1 else 's')
    if errors:
        line += ' / %d error%s' % (errors, 's' if errors != 1 else '')
    if deselected:
        line += ' / %d deselected' % deselected
    if skipped:
        line += ' / %d skipped' % skipped
    if self._numcollected > selected:
        line += ' / %d selected' % selected
    if self.isatty:
        self.rewrite(line, bold=True, erase=True)
        if final:
            self.write('\n')
    else:
        self.write_line(line)