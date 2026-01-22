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
def debug_data_file(filename: str) -> None:
    """Implementation of 'coverage debug data'."""
    data = CoverageData(filename)
    filename = data.data_filename()
    print(f'path: {filename}')
    if not os.path.exists(filename):
        print("No data collected: file doesn't exist")
        return
    data.read()
    print(f'has_arcs: {data.has_arcs()!r}')
    summary = line_counts(data, fullpath=True)
    filenames = human_sorted(summary.keys())
    nfiles = len(filenames)
    print(f'{nfiles} file{plural(nfiles)}:')
    for f in filenames:
        line = f'{f}: {summary[f]} line{plural(summary[f])}'
        plugin = data.file_tracer(f)
        if plugin:
            line += f' [{plugin}]'
        print(line)