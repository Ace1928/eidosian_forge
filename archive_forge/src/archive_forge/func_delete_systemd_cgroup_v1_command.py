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
@property
def delete_systemd_cgroup_v1_command(self) -> list[str]:
    """The command used to remove the previously created ansible-test cgroup in the v1 systemd hierarchy."""
    return ['find', self.cgroup_path, '-type', 'd', '-delete']