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
def run_ast_checks(self) -> None:
    """Run all checks expecting an abstract syntax tree."""
    assert self.processor is not None, self.filename
    ast = self.processor.build_ast()
    for plugin in self.plugins.tree:
        checker = self.run_check(plugin, tree=ast)
        try:
            runner = checker.run()
        except AttributeError:
            runner = checker
        for line_number, offset, text, _ in runner:
            self.report(error_code=None, line_number=line_number, column=offset, text=text)