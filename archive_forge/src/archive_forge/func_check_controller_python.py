from __future__ import annotations
import collections.abc as c
import dataclasses
import functools
import itertools
import os
import pickle
import sys
import time
import traceback
import typing as t
from .config import (
from .util import (
from .util_common import (
from .thread import (
from .host_profiles import (
from .pypi_proxy import (
def check_controller_python(args: EnvironmentConfig, host_state: HostState) -> None:
    """Check the running environment to make sure it is what we expected."""
    sys_version = version_to_str(sys.version_info[:2])
    controller_python = host_state.controller_profile.python
    if (expected_executable := verify_sys_executable(controller_python.path)):
        raise ApplicationError(f'Running under Python interpreter "{sys.executable}" instead of "{expected_executable}".')
    expected_version = controller_python.version
    if expected_version != sys_version:
        raise ApplicationError(f'Running under Python version {sys_version} instead of {expected_version}.')
    args.controller_python = controller_python