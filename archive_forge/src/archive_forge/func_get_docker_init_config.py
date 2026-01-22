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
def get_docker_init_config(self) -> InitConfig:
    """Return init config for running under Docker."""
    options = self.get_common_run_options()
    command: t.Optional[str] = None
    command_privileged = False
    expected_mounts: tuple[CGroupMount, ...]
    cgroup_version = get_docker_info(self.args).cgroup_version
    if self.config.cgroup == CGroupVersion.NONE:
        if get_docker_info(self.args).cgroupns_option_supported:
            options.extend(('--cgroupns', 'private'))
        options.extend(('--tmpfs', '/sys/fs/cgroup'))
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None),)
    elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V1_ONLY) and cgroup_version == 1:
        if get_docker_info(self.args).cgroupns_option_supported:
            options.extend(('--cgroupns', 'host'))
        options.extend(('--tmpfs', '/sys/fs/cgroup', '--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw'))
        self.check_systemd_cgroup_v1(options)
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
    elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V2_ONLY) and cgroup_version == 2:
        command = 'mount -o remount,rw /sys/fs/cgroup/'
        command_privileged = True
        options.extend(('--cgroupns', 'private'))
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE),)
    elif self.config.cgroup == CGroupVersion.V1_ONLY and cgroup_version == 2:
        cgroup_path = self.create_systemd_cgroup_v1()
        command = f'echo 1 > {cgroup_path}/cgroup.procs'
        options.extend(('--cgroupns', 'private', '--tmpfs', '/sys/fs/cgroup', '--tmpfs', '/sys/fs/cgroup/systemd', '--volume', f'{cgroup_path}:{cgroup_path}:rw'))
        expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=cgroup_path, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
    else:
        raise InternalError(f'Unhandled cgroup configuration: {self.config.cgroup} on cgroup v{cgroup_version}.')
    return self.InitConfig(options=options, command=command, command_privileged=command_privileged, expected_mounts=expected_mounts)