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
def check_cgroup_requirements(self) -> None:
    """Check cgroup requirements for the container."""
    cgroup_version = get_docker_info(self.args).cgroup_version
    if cgroup_version not in (1, 2):
        raise ApplicationError(f'The container host provides cgroup v{cgroup_version}, but only version v1 and v2 are supported.')
    if self.config.cgroup == CGroupVersion.V2_ONLY and cgroup_version != 2:
        raise ApplicationError(f'Container {self.config.name} requires cgroup v2 but the container host provides cgroup v{cgroup_version}.')
    if self.config.cgroup == CGroupVersion.V1_ONLY or (self.config.cgroup != CGroupVersion.NONE and get_docker_info(self.args).cgroup_version == 1):
        if (cgroup_v1 := detect_host_properties(self.args).cgroup_v1) != SystemdControlGroupV1Status.VALID:
            if self.config.cgroup == CGroupVersion.V1_ONLY:
                if get_docker_info(self.args).cgroup_version == 2:
                    reason = f'Container {self.config.name} requires cgroup v1, but the container host only provides cgroup v2.'
                else:
                    reason = f'Container {self.config.name} requires cgroup v1, but the container host does not appear to be running systemd.'
            else:
                reason = 'The container host provides cgroup v1, but does not appear to be running systemd.'
            reason += f'\n{cgroup_v1.value}'
            raise ControlGroupError(self.args, reason)