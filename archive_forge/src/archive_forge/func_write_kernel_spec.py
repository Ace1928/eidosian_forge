from __future__ import annotations
import errno
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any
from jupyter_client.kernelspec import KernelSpecManager
from traitlets import Unicode
from traitlets.config import Application
def write_kernel_spec(path: Path | str | None=None, overrides: dict[str, Any] | None=None, extra_arguments: list[str] | None=None, python_arguments: list[str] | None=None) -> str:
    """Write a kernel spec directory to `path`

    If `path` is not specified, a temporary directory is created.
    If `overrides` is given, the kernelspec JSON is updated before writing.

    The path to the kernelspec is always returned.
    """
    if path is None:
        path = Path(tempfile.mkdtemp(suffix='_kernels')) / KERNEL_NAME
    shutil.copytree(RESOURCES, path)
    mask = Path(path).stat().st_mode
    if not mask & stat.S_IWUSR:
        Path(path).chmod(mask | stat.S_IWUSR)
    kernel_dict = get_kernel_dict(extra_arguments, python_arguments)
    if overrides:
        kernel_dict.update(overrides)
    with open(pjoin(path, 'kernel.json'), 'w') as f:
        json.dump(kernel_dict, f, indent=1)
    return str(path)