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
def iter_scripts_to_remove(dist: BaseDistribution, bin_dir: str) -> Generator[str, None, None]:
    for entry_point in dist.iter_entry_points():
        if entry_point.group == 'console_scripts':
            yield from _script_names(bin_dir, entry_point.name, False)
        elif entry_point.group == 'gui_scripts':
            yield from _script_names(bin_dir, entry_point.name, True)