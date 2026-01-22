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
def combinable_files(data_file: str, data_paths: Iterable[str] | None=None) -> list[str]:
    """Make a list of data files to be combined.

    `data_file` is a path to a data file.  `data_paths` is a list of files or
    directories of files.

    Returns a list of absolute file paths.
    """
    data_dir, local = os.path.split(os.path.abspath(data_file))
    data_paths = data_paths or [data_dir]
    files_to_combine = []
    for p in data_paths:
        if os.path.isfile(p):
            files_to_combine.append(os.path.abspath(p))
        elif os.path.isdir(p):
            pattern = glob.escape(os.path.join(os.path.abspath(p), local)) + '.*'
            files_to_combine.extend(glob.glob(pattern))
        else:
            raise NoDataError(f"Couldn't combine from non-existent path '{p}'")
    files_to_combine = [fnm for fnm in files_to_combine if not fnm.endswith('-journal')]
    return sorted(files_to_combine)