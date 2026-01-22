from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import (
from .io import (
from .util import (
from .data import (
from .util_common import (
from .host_configs import (
from .constants import (
from .thread import (
def get_coverage_platform(config: HostConfig) -> str:
    """Return the platform label for the given host config."""
    if isinstance(config, PosixRemoteConfig):
        platform = f'remote-{sanitize_host_name(config.name)}'
    elif isinstance(config, DockerConfig):
        platform = f'docker-{sanitize_host_name(config.name)}'
    elif isinstance(config, PosixSshConfig):
        platform = f'ssh-{sanitize_host_name(config.host)}'
    elif isinstance(config, OriginConfig):
        platform = 'origin'
    else:
        raise NotImplementedError(f'Coverage platform label not defined for type: {type(config)}')
    return platform