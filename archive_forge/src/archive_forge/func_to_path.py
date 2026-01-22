import os
import platform
from functools import wraps
from pathlib import PurePath, PurePosixPath
from typing import Any, NewType, Union
def to_path(self) -> PurePosixPath:
    """Convert this path to a PurePosixPath."""
    return PurePosixPath(self)