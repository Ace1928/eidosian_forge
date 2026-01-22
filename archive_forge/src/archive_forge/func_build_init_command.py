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
def build_init_command(self, init_config: InitConfig, sleep: bool) -> t.Optional[list[str]]:
    """
        Build and return the command to start in the container.
        Returns None if the default command for the container should be used.

        The sleep duration below was selected to:

          - Allow enough time to perform necessary operations in the container before waking it.
          - Make the delay obvious if the wake command doesn't run or succeed.
          - Avoid hanging indefinitely or for an unreasonably long time.

        NOTE: The container must have a POSIX-compliant default shell "sh" with a non-builtin "sleep" command.
              The "sleep" command is invoked through "env" to avoid using a shell builtin "sleep" (if present).
        """
    command = ''
    if init_config.command and (not init_config.command_privileged):
        command += f'{init_config.command} && '
    if sleep or init_config.command_privileged:
        command += 'env sleep 60 ; '
    if not command:
        return None
    docker_pull(self.args, self.config.image)
    inspect = docker_image_inspect(self.args, self.config.image)
    command += f'exec {shlex.join(inspect.cmd)}'
    return ['sh', '-c', command]