import errno
import functools
import os
import socket
import subprocess
import sys
from collections import namedtuple
from socket import AF_INET
from . import _common
from . import _psposix
from . import _psutil_posix as cext_posix
from . import _psutil_sunos as cext
from ._common import AF_INET6
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import debug
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import sockfam_to_enum
from ._common import socktype_to_enum
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
def oneshot_enter(self):
    self._proc_name_and_args.cache_activate(self)
    self._proc_basic_info.cache_activate(self)
    self._proc_cred.cache_activate(self)