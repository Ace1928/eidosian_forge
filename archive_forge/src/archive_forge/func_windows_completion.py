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
def windows_completion() -> dict[str, WindowsRemoteCompletionConfig]:
    """Return windows completion entries."""
    return load_completion('windows', WindowsRemoteCompletionConfig)