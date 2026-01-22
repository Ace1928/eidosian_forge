from __future__ import annotations
import abc
import shlex
from .util import (
def prepare_command(self, command: list[str]) -> list[str]:
    """Return the given command, if any, with privilege escalation."""
    become = ['sudo', '-in']
    if command:
        become.extend(['sh', '-c', shlex.join(command)])
    return become