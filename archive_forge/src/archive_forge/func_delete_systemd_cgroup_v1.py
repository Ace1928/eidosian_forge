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
def delete_systemd_cgroup_v1(self) -> None:
    """Delete a previously created ansible-test cgroup in the v1 systemd hierarchy."""
    options = ['--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw', '--privileged']
    cmd = ['sh', '-c', f'>&2 echo {shlex.quote(self.MARKER)} && {shlex.join(self.delete_systemd_cgroup_v1_command)}']
    try:
        run_utility_container(self.args, f'ansible-test-cgroup-delete-{self.label}', cmd, options)
    except SubprocessError as ex:
        if (error := self.extract_error(ex.stderr)):
            if error.endswith(': No such file or directory'):
                return
        display.error(str(ex))