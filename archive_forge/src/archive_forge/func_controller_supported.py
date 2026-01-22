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
@property
def controller_supported(self) -> bool:
    """True if at least one Python version is provided which supports the controller, otherwise False."""
    return any((version in CONTROLLER_PYTHON_VERSIONS for version in self.supported_pythons))