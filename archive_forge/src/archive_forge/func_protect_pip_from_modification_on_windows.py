import contextlib
import errno
import getpass
import hashlib
import io
import logging
import os
import posixpath
import shutil
import stat
import sys
import sysconfig
import urllib.parse
from functools import partial
from io import StringIO
from itertools import filterfalse, tee, zip_longest
from pathlib import Path
from types import FunctionType, TracebackType
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip import __version__
from pip._internal.exceptions import CommandError, ExternallyManagedEnvironment
from pip._internal.locations import get_major_minor_version
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
def protect_pip_from_modification_on_windows(modifying_pip: bool) -> None:
    """Protection of pip.exe from modification on Windows

    On Windows, any operation modifying pip should be run as:
        python -m pip ...
    """
    pip_names = ['pip', f'pip{sys.version_info.major}', f'pip{sys.version_info.major}.{sys.version_info.minor}']
    should_show_use_python_msg = modifying_pip and WINDOWS and (os.path.basename(sys.argv[0]) in pip_names)
    if should_show_use_python_msg:
        new_command = [sys.executable, '-m', 'pip'] + sys.argv[1:]
        raise CommandError('To modify pip, please run the following command:\n{}'.format(' '.join(new_command)))