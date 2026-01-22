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
def make_ipkernel_cmd(mod: str='ipykernel_launcher', executable: str | None=None, extra_arguments: list[str] | None=None, python_arguments: list[str] | None=None) -> list[str]:
    """Build Popen command list for launching an IPython kernel.

    Parameters
    ----------
    mod : str, optional (default 'ipykernel')
        A string of an IPython module whose __main__ starts an IPython kernel
    executable : str, optional (default sys.executable)
        The Python executable to use for the kernel process.
    extra_arguments : list, optional
        A list of extra arguments to pass when executing the launch code.

    Returns
    -------
    A Popen command list
    """
    if executable is None:
        executable = sys.executable
    extra_arguments = extra_arguments or []
    python_arguments = python_arguments or []
    return [executable, *python_arguments, '-m', mod, '-f', '{connection_file}', *extra_arguments]