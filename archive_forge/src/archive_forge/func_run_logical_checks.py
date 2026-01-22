from __future__ import annotations
import argparse
import contextlib
import errno
import logging
import multiprocessing.pool
import operator
import signal
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from flake8 import defaults
from flake8 import exceptions
from flake8 import processor
from flake8 import utils
from flake8._compat import FSTRING_START
from flake8.discover_files import expand_paths
from flake8.options.parse_args import parse_args
from flake8.plugins.finder import Checkers
from flake8.plugins.finder import LoadedPlugin
from flake8.style_guide import StyleGuideManager
def run_logical_checks(self) -> None:
    """Run all checks expecting a logical line."""
    assert self.processor is not None
    comments, logical_line, mapping = self.processor.build_logical_line()
    if not mapping:
        return
    self.processor.update_state(mapping)
    LOG.debug('Logical line: "%s"', logical_line.rstrip())
    for plugin in self.plugins.logical_line:
        self.processor.update_checker_state_for(plugin)
        results = self.run_check(plugin, logical_line=logical_line) or ()
        for offset, text in results:
            line_number, column_offset = find_offset(offset, mapping)
            if line_number == column_offset == 0:
                LOG.warning('position of error out of bounds: %s', plugin)
            self.report(error_code=None, line_number=line_number, column=column_offset, text=text)
    self.processor.next_logical_line()