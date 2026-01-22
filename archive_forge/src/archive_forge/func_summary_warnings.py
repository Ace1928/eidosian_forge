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
def summary_warnings(self) -> None:
    if self.hasopt('w'):
        all_warnings: Optional[List[WarningReport]] = self.stats.get('warnings')
        if not all_warnings:
            return
        final = self._already_displayed_warnings is not None
        if final:
            warning_reports = all_warnings[self._already_displayed_warnings:]
        else:
            warning_reports = all_warnings
        self._already_displayed_warnings = len(warning_reports)
        if not warning_reports:
            return
        reports_grouped_by_message: Dict[str, List[WarningReport]] = {}
        for wr in warning_reports:
            reports_grouped_by_message.setdefault(wr.message, []).append(wr)

        def collapsed_location_report(reports: List[WarningReport]) -> str:
            locations = []
            for w in reports:
                location = w.get_location(self.config)
                if location:
                    locations.append(location)
            if len(locations) < 10:
                return '\n'.join(map(str, locations))
            counts_by_filename = Counter((str(loc).split('::', 1)[0] for loc in locations))
            return '\n'.join(('{}: {} warning{}'.format(k, v, 's' if v > 1 else '') for k, v in counts_by_filename.items()))
        title = 'warnings summary (final)' if final else 'warnings summary'
        self.write_sep('=', title, yellow=True, bold=False)
        for message, message_reports in reports_grouped_by_message.items():
            maybe_location = collapsed_location_report(message_reports)
            if maybe_location:
                self._tw.line(maybe_location)
                lines = message.splitlines()
                indented = '\n'.join(('  ' + x for x in lines))
                message = indented.rstrip()
            else:
                message = message.rstrip()
            self._tw.line(message)
            self._tw.line()
        self._tw.line('-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html')