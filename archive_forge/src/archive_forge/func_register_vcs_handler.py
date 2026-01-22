import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def register_vcs_handler(vcs, method):
    """Create decorator to mark a method as the handler of a VCS."""

    def decorate(f):
        """Store f in HANDLERS[vcs][method]."""
        if vcs not in HANDLERS:
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate