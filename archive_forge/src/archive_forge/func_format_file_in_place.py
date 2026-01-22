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
def format_file_in_place(src: Path, fast: bool, mode: Mode, write_back: WriteBack=WriteBack.NO, lock: Any=None, *, lines: Collection[Tuple[int, int]]=()) -> bool:
    """Format file under `src` path. Return True if changed.

    If `write_back` is DIFF, write a diff to stdout. If it is YES, write reformatted
    code to the file.
    `mode` and `fast` options are passed to :func:`format_file_contents`.
    """
    if src.suffix == '.pyi':
        mode = replace(mode, is_pyi=True)
    elif src.suffix == '.ipynb':
        mode = replace(mode, is_ipynb=True)
    then = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc)
    header = b''
    with open(src, 'rb') as buf:
        if mode.skip_source_first_line:
            header = buf.readline()
        src_contents, encoding, newline = decode_bytes(buf.read())
    try:
        dst_contents = format_file_contents(src_contents, fast=fast, mode=mode, lines=lines)
    except NothingChanged:
        return False
    except JSONDecodeError:
        raise ValueError(f"File '{src}' cannot be parsed as valid Jupyter notebook.") from None
    src_contents = header.decode(encoding) + src_contents
    dst_contents = header.decode(encoding) + dst_contents
    if write_back == WriteBack.YES:
        with open(src, 'w', encoding=encoding, newline=newline) as f:
            f.write(dst_contents)
    elif write_back in (WriteBack.DIFF, WriteBack.COLOR_DIFF):
        now = datetime.now(timezone.utc)
        src_name = f'{src}\t{then}'
        dst_name = f'{src}\t{now}'
        if mode.is_ipynb:
            diff_contents = ipynb_diff(src_contents, dst_contents, src_name, dst_name)
        else:
            diff_contents = diff(src_contents, dst_contents, src_name, dst_name)
        if write_back == WriteBack.COLOR_DIFF:
            diff_contents = color_diff(diff_contents)
        with lock or nullcontext():
            f = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding, newline=newline, write_through=True)
            f = wrap_stream_for_windows(f)
            f.write(diff_contents)
            f.detach()
    return True