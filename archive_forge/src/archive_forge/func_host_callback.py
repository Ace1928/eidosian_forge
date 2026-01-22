from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
def host_callback(payload_config: PayloadConfig) -> None:
    """Add the host files to the payload file list."""
    config = self
    if config.host_path:
        settings_path = os.path.join(config.host_path, 'settings.dat')
        state_path = os.path.join(config.host_path, 'state.dat')
        config_path = os.path.join(config.host_path, 'config.dat')
        files = payload_config.files
        files.append((os.path.abspath(settings_path), settings_path))
        files.append((os.path.abspath(state_path), state_path))
        files.append((os.path.abspath(config_path), config_path))