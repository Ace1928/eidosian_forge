from __future__ import annotations
import re
import typing as t
from .util import (
def get_submodule_paths(self) -> list[str]:
    """Return a list of submodule paths recursively."""
    cmd = ['submodule', 'status', '--recursive']
    output = self.run_git_split(cmd, '\n')
    submodule_paths = [re.search('^.[0-9a-f]+ (?P<path>[^ ]+)', line).group('path') for line in output]
    submodule_paths = [path for path in submodule_paths if not path.startswith('../')]
    return submodule_paths