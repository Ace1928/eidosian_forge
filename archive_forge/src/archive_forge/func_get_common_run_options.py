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
def get_common_run_options(self) -> list[str]:
    """Return a list of options needed to run the container."""
    options = ['--tmpfs', '/tmp:exec', '--tmpfs', '/run:exec', '--tmpfs', '/run/lock']
    if self.config.privileged:
        options.append('--privileged')
    if self.config.memory:
        options.extend([f'--memory={self.config.memory}', f'--memory-swap={self.config.memory}'])
    if self.config.seccomp != 'default':
        options.extend(['--security-opt', f'seccomp={self.config.seccomp}'])
    docker_socket = '/var/run/docker.sock'
    if get_docker_hostname() != 'localhost' or os.path.exists(docker_socket):
        options.extend(['--volume', f'{docker_socket}:{docker_socket}'])
    return options