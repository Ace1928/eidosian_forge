from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import (
from .config import (
from .host_configs import (
from .core_ci import (
from .util import (
from .util_common import (
from .docker_util import (
from .bootstrap import (
from .venv import (
from .ssh import (
from .ansible_util import (
from .containers import (
from .connections import (
from .become import (
from .completion import (
from .dev.container_probe import (
def get_controller_target_connections(self) -> list[SshConnection]:
    """Return SSH connection(s) for accessing the host as a target from the controller."""
    core_ci = self.wait_for_instance()
    settings = SshConnectionDetail(name=core_ci.name, host=core_ci.connection.hostname, port=22, user=core_ci.connection.username, identity_file=core_ci.ssh_key.key, shell_type='powershell')
    return [SshConnection(self.args, settings)]