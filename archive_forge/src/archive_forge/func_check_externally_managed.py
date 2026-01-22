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
def check_externally_managed() -> None:
    """Check whether the current environment is externally managed.

    If the ``EXTERNALLY-MANAGED`` config file is found, the current environment
    is considered externally managed, and an ExternallyManagedEnvironment is
    raised.
    """
    if running_under_virtualenv():
        return
    marker = os.path.join(sysconfig.get_path('stdlib'), 'EXTERNALLY-MANAGED')
    if not os.path.isfile(marker):
        return
    raise ExternallyManagedEnvironment.from_config(marker)