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
def get_inventory_variables(self) -> dict[str, t.Optional[t.Union[str, int]]]:
    """Return inventory variables for accessing this host."""
    core_ci = self.wait_for_instance()
    connection = core_ci.connection
    variables: dict[str, t.Optional[t.Union[str, int]]] = dict(ansible_connection='winrm', ansible_pipelining='yes', ansible_winrm_server_cert_validation='ignore', ansible_host=connection.hostname, ansible_port=connection.port, ansible_user=connection.username, ansible_password=connection.password, ansible_ssh_private_key_file=core_ci.ssh_key.key)
    if self.config.version == '2016':
        variables.update(ansible_winrm_transport='ntlm', ansible_winrm_scheme='http', ansible_port='5985')
    return variables