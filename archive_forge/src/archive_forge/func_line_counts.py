imports working.
from __future__ import annotations
import glob
import hashlib
import os.path
from typing import Callable, Iterable
from coverage.exceptions import CoverageException, NoDataError
from coverage.files import PathAliases
from coverage.misc import Hasher, file_be_gone, human_sorted, plural
from coverage.sqldata import CoverageData
def line_counts(data: CoverageData, fullpath: bool=False) -> dict[str, int]:
    """Return a dict summarizing the line coverage data.

    Keys are based on the file names, and values are the number of executed
    lines.  If `fullpath` is true, then the keys are the full pathnames of
    the files, otherwise they are the basenames of the files.

    Returns a dict mapping file names to counts of lines.

    """
    summ = {}
    filename_fn: Callable[[str], str]
    if fullpath:
        filename_fn = lambda f: f
    else:
        filename_fn = os.path.basename
    for filename in data.measured_files():
        lines = data.lines(filename)
        assert lines is not None
        summ[filename_fn(filename)] = len(lines)
    return summ