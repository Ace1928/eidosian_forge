from __future__ import annotations
import re
import typing as t
from .util import (
def run_git_split(self, cmd: list[str], separator: t.Optional[str]=None, str_errors: str='strict') -> list[str]:
    """Run the given `git` command and return the results as a list."""
    output = self.run_git(cmd, str_errors=str_errors).strip(separator)
    if not output:
        return []
    return output.split(separator)