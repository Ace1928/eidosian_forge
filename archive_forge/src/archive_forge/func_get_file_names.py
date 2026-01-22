from __future__ import annotations
import re
import typing as t
from .util import (
def get_file_names(self, args: list[str]) -> list[str]:
    """Return a list of file names from the `git ls-files` command."""
    cmd = ['ls-files', '-z'] + args
    return self.run_git_split(cmd, '\x00')