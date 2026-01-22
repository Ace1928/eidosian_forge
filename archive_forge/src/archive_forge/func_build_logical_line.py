from __future__ import annotations
import argparse
import ast
import functools
import logging
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Tuple
from flake8 import defaults
from flake8 import utils
from flake8._compat import FSTRING_END
from flake8._compat import FSTRING_MIDDLE
from flake8.plugins.finder import LoadedPlugin
def build_logical_line(self) -> tuple[str, str, _LogicalMapping]:
    """Build a logical line from the current tokens list."""
    comments, logical, mapping_list = self.build_logical_line_tokens()
    joined_comments = ''.join(comments)
    self.logical_line = ''.join(logical)
    self.statistics['logical lines'] += 1
    return (joined_comments, self.logical_line, mapping_list)