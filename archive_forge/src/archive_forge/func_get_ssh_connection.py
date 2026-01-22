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
def get_ssh_connection(self) -> SshConnection:
    """Return an SSH connection for accessing the host."""
    core_ci = self.wait_for_instance()
    settings = SshConnectionDetail(name=core_ci.name, user=core_ci.connection.username, host=core_ci.connection.hostname, port=core_ci.connection.port, identity_file=core_ci.ssh_key.key, python_interpreter=self.python.path)
    if settings.user == 'root':
        become: t.Optional[Become] = None
    elif self.config.become:
        become = SUPPORTED_BECOME_METHODS[self.config.become]()
    else:
        display.warning(f'Defaulting to "sudo" for platform "{self.config.platform}" become support.', unique=True)
        become = Sudo()
    return SshConnection(self.args, settings, become)