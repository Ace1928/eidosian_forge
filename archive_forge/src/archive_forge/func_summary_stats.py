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
def summary_stats(self) -> None:
    if self.verbosity < -1:
        return
    session_duration = timing.time() - self._sessionstarttime
    parts, main_color = self.build_summary_stats_line()
    line_parts = []
    display_sep = self.verbosity >= 0
    if display_sep:
        fullwidth = self._tw.fullwidth
    for text, markup in parts:
        with_markup = self._tw.markup(text, **markup)
        if display_sep:
            fullwidth += len(with_markup) - len(text)
        line_parts.append(with_markup)
    msg = ', '.join(line_parts)
    main_markup = {main_color: True}
    duration = f' in {format_session_duration(session_duration)}'
    duration_with_markup = self._tw.markup(duration, **main_markup)
    if display_sep:
        fullwidth += len(duration_with_markup) - len(duration)
    msg += duration_with_markup
    if display_sep:
        markup_for_end_sep = self._tw.markup('', **main_markup)
        if markup_for_end_sep.endswith('\x1b[0m'):
            markup_for_end_sep = markup_for_end_sep[:-4]
        fullwidth += len(markup_for_end_sep)
        msg += markup_for_end_sep
    if display_sep:
        self.write_sep('=', msg, fullwidth=fullwidth, **main_markup)
    else:
        self.write_line(msg, **main_markup)