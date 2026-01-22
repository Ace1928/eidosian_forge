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
def short_test_summary(self) -> None:
    if not self.reportchars:
        return

    def show_simple(lines: List[str], *, stat: str) -> None:
        failed = self.stats.get(stat, [])
        if not failed:
            return
        config = self.config
        for rep in failed:
            color = _color_for_type.get(stat, _color_for_type_default)
            line = _get_line_with_reprcrash_message(config, rep, self._tw, {color: True})
            lines.append(line)

    def show_xfailed(lines: List[str]) -> None:
        xfailed = self.stats.get('xfailed', [])
        for rep in xfailed:
            verbose_word = rep._get_verbose_word(self.config)
            markup_word = self._tw.markup(verbose_word, **{_color_for_type['warnings']: True})
            nodeid = _get_node_id_with_markup(self._tw, self.config, rep)
            line = f'{markup_word} {nodeid}'
            reason = rep.wasxfail
            if reason:
                line += ' - ' + str(reason)
            lines.append(line)

    def show_xpassed(lines: List[str]) -> None:
        xpassed = self.stats.get('xpassed', [])
        for rep in xpassed:
            verbose_word = rep._get_verbose_word(self.config)
            markup_word = self._tw.markup(verbose_word, **{_color_for_type['warnings']: True})
            nodeid = _get_node_id_with_markup(self._tw, self.config, rep)
            line = f'{markup_word} {nodeid}'
            reason = rep.wasxfail
            if reason:
                line += ' - ' + str(reason)
            lines.append(line)

    def show_skipped(lines: List[str]) -> None:
        skipped: List[CollectReport] = self.stats.get('skipped', [])
        fskips = _folded_skips(self.startpath, skipped) if skipped else []
        if not fskips:
            return
        verbose_word = skipped[0]._get_verbose_word(self.config)
        markup_word = self._tw.markup(verbose_word, **{_color_for_type['warnings']: True})
        prefix = 'Skipped: '
        for num, fspath, lineno, reason in fskips:
            if reason.startswith(prefix):
                reason = reason[len(prefix):]
            if lineno is not None:
                lines.append('%s [%d] %s:%d: %s' % (markup_word, num, fspath, lineno, reason))
            else:
                lines.append('%s [%d] %s: %s' % (markup_word, num, fspath, reason))
    REPORTCHAR_ACTIONS: Mapping[str, Callable[[List[str]], None]] = {'x': show_xfailed, 'X': show_xpassed, 'f': partial(show_simple, stat='failed'), 's': show_skipped, 'p': partial(show_simple, stat='passed'), 'E': partial(show_simple, stat='error')}
    lines: List[str] = []
    for char in self.reportchars:
        action = REPORTCHAR_ACTIONS.get(char)
        if action:
            action(lines)
    if lines:
        self.write_sep('=', 'short test summary info', cyan=True, bold=True)
        for line in lines:
            self.write_line(line)