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
def deprovision(self) -> None:
    """Deprovision the host after delegation has completed."""
    container_exists = False
    if self.container_name:
        if self.args.docker_terminate == TerminateMode.ALWAYS or (self.args.docker_terminate == TerminateMode.SUCCESS and self.args.success):
            docker_rm(self.args, self.container_name)
        else:
            container_exists = True
    if self.cgroup_path:
        if container_exists:
            display.notice(f'Remember to run `{require_docker().command} rm -f {self.container_name}` when finished testing. Then run `{shlex.join(self.delete_systemd_cgroup_v1_command)}` on the container host.')
        else:
            self.delete_systemd_cgroup_v1()
    elif container_exists:
        display.notice(f'Remember to run `{require_docker().command} rm -f {self.container_name}` when finished testing.')