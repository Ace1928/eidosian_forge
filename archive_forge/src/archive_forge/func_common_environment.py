from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def common_environment() -> dict[str, str]:
    """Common environment used for executing all programs."""
    env = dict(LC_ALL=CONFIGURED_LOCALE, PATH=os.environ.get('PATH', os.path.defpath))
    required = ('HOME',)
    optional = ('LD_LIBRARY_PATH', 'SSH_AUTH_SOCK', 'OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'ANSIBLE_KEEP_REMOTE_FILES', 'LDFLAGS', 'CFLAGS')
    if os.path.exists('/etc/freebsd-update.conf'):
        env.update(CFLAGS='-I/usr/local/include/')
    env.update(pass_vars(required=required, optional=optional))
    return env