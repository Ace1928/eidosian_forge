from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
def get_cli_path(path: str) -> str:
    """Return the absolute path to the CLI script from the given path which is relative to the `bin` directory of the original source tree layout."""
    path_rewrite = {'../lib/ansible/': ANSIBLE_LIB_ROOT, '../test/lib/ansible_test/': ANSIBLE_TEST_ROOT}
    for prefix, destination in path_rewrite.items():
        if path.startswith(prefix):
            return os.path.join(destination, path[len(prefix):])
    raise RuntimeError(path)