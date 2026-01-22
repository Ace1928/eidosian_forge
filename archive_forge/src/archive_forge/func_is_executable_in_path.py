from __future__ import annotations
import os
import platform
import re
import sys
def is_executable_in_path(name: str) -> bool:
    """Check if executable is in OS path."""
    from shutil import which
    return which(name) is not None