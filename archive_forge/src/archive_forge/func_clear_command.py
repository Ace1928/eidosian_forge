from __future__ import annotations
import enum
import typing
from typing import Iterator
def clear_command(self, command: str | Command) -> None:
    dk = [k for k, v in self._command.items() if v == command]
    for k in dk:
        del self._command[k]