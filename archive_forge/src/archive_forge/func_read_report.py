from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def read_report(path: str) -> tuple[list[str], Arcs, Lines]:
    """Read a JSON report from disk."""
    try:
        report = read_json_file(path)
    except Exception as ex:
        raise ApplicationError('File "%s" is not valid JSON: %s' % (path, ex)) from None
    try:
        return load_report(report)
    except ApplicationError as ex:
        raise ApplicationError('File "%s" is not an aggregated coverage data file. %s' % (path, ex)) from None