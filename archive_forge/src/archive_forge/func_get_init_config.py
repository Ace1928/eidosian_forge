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
def get_init_config(self) -> InitConfig:
    """Return init config for running under the current container engine."""
    self.check_cgroup_requirements()
    engine = require_docker().command
    init_config = getattr(self, f'get_{engine}_init_config')()
    return init_config