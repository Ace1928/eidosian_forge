from __future__ import annotations
import collections.abc as c
import contextlib
import json
import random
import time
import uuid
import threading
import typing as t
from .util import (
from .util_common import (
from .config import (
from .docker_util import (
from .ansible_util import (
from .core_ci import (
from .target import (
from .ssh import (
from .host_configs import (
from .connections import (
from .thread import (
def local_ssh(args: EnvironmentConfig, python: PythonConfig) -> SshConnectionDetail:
    """Return SSH connection details for localhost, connecting as root to the default SSH port."""
    return SshConnectionDetail('localhost', 'localhost', None, 'root', SshKey(args).key, python.path)