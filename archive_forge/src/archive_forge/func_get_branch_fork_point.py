from __future__ import annotations
import re
import typing as t
from .util import (
def get_branch_fork_point(self, branch: str) -> str:
    """Return a reference to the point at which the given branch was forked."""
    cmd = ['merge-base', branch, 'HEAD']
    return self.run_git(cmd).strip()