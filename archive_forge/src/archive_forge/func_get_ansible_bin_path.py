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
@mutex
def get_ansible_bin_path(args: CommonConfig) -> str:
    """
    Return a directory usable for PATH, containing only the ansible entry points.
    If a temporary directory is required, it will be cached for the lifetime of the process and cleaned up at exit.
    """
    try:
        return get_ansible_bin_path.bin_path
    except AttributeError:
        pass
    if ANSIBLE_SOURCE_ROOT:
        bin_path = os.path.join(ANSIBLE_ROOT, 'bin')
    else:
        bin_path = create_temp_dir(prefix='ansible-test-', suffix='-bin')
        bin_links = {os.path.join(bin_path, name): get_cli_path(path) for name, path in ANSIBLE_BIN_SYMLINK_MAP.items()}
        if not args.explain:
            for dst, src in bin_links.items():
                shutil.copy(src, dst)
                verified_chmod(dst, MODE_FILE_EXECUTE)
    get_ansible_bin_path.bin_path = bin_path
    return bin_path