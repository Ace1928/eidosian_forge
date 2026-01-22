from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@cache
def docker_completion() -> dict[str, DockerCompletionConfig]:
    """Return docker completion entries."""
    return load_completion('docker', DockerCompletionConfig)