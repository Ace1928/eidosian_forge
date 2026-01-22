import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
def compress_for_output_listing(paths: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """Returns a tuple of 2 sets of which paths to display to user

    The first set contains paths that would be deleted. Files of a package
    are not added and the top-level directory of the package has a '*' added
    at the end - to signify that all it's contents are removed.

    The second set contains files that would have been skipped in the above
    folders.
    """
    will_remove = set(paths)
    will_skip = set()
    folders = set()
    files = set()
    for path in will_remove:
        if path.endswith('.pyc'):
            continue
        if path.endswith('__init__.py') or '.dist-info' in path:
            folders.add(os.path.dirname(path))
        files.add(path)
    _normcased_files = set(map(os.path.normcase, files))
    folders = compact(folders)
    for folder in folders:
        for dirpath, _, dirfiles in os.walk(folder):
            for fname in dirfiles:
                if fname.endswith('.pyc'):
                    continue
                file_ = os.path.join(dirpath, fname)
                if os.path.isfile(file_) and os.path.normcase(file_) not in _normcased_files:
                    will_skip.add(file_)
    will_remove = files | {os.path.join(folder, '*') for folder in folders}
    return (will_remove, will_skip)