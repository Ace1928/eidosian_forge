from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
@property
def has_python(self) -> bool:
    """Return True if this mode uses Python, otherwise False."""
    return self in (TargetMode.POSIX_INTEGRATION, TargetMode.SANITY, TargetMode.UNITS, TargetMode.SHELL)