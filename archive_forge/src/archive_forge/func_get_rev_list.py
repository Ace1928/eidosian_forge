from __future__ import annotations
import re
import typing as t
from .util import (
def get_rev_list(self, commits: t.Optional[list[str]]=None, max_count: t.Optional[int]=None) -> list[str]:
    """Return the list of results from the `git rev-list` command."""
    cmd = ['rev-list']
    if commits:
        cmd += commits
    else:
        cmd += ['HEAD']
    if max_count:
        cmd += ['--max-count', '%s' % max_count]
    return self.run_git_split(cmd)