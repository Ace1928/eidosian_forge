import io
import json
import platform
import re
import sys
import tokenize
import traceback
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import (
import click
from click.core import ParameterSource
from mypy_extensions import mypyc_attr
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from _black_version import version as __version__
from black.cache import Cache
from black.comments import normalize_fmt_off
from black.const import (
from black.files import (
from black.handle_ipynb_magics import (
from black.linegen import LN, LineGenerator, transform_line
from black.lines import EmptyLineTracker, LinesBlock
from black.mode import FUTURE_FLAG_TO_FEATURE, VERSION_TO_FEATURES, Feature
from black.mode import Mode as Mode  # re-exported
from black.mode import Preview, TargetVersion, supports_feature
from black.nodes import (
from black.output import color_diff, diff, dump_to_file, err, ipynb_diff, out
from black.parsing import (  # noqa F401
from black.ranges import (
from black.report import Changed, NothingChanged, Report
from black.trans import iter_fexpr_spans
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def assert_stable(src: str, dst: str, mode: Mode, *, lines: Collection[Tuple[int, int]]=()) -> None:
    """Raise AssertionError if `dst` reformats differently the second time."""
    if lines:
        return
    newdst = _format_str_once(dst, mode=mode, lines=lines)
    if dst != newdst:
        log = dump_to_file(str(mode), diff(src, dst, 'source', 'first pass'), diff(dst, newdst, 'first pass', 'second pass'))
        raise AssertionError(f'INTERNAL ERROR: Black produced different code on the second pass of the formatter.  Please report a bug on https://github.com/psf/black/issues.  This diff might be helpful: {log}') from None