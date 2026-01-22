from __future__ import annotations
import argparse
import errno
import json
import os
import site
import sys
import sysconfig
from pathlib import Path
from shutil import which
from subprocess import Popen
from typing import Any
from . import paths
from .version import __version__
def list_subcommands() -> list[str]:
    """List all jupyter subcommands

    searches PATH for `jupyter-name`

    Returns a list of jupyter's subcommand names, without the `jupyter-` prefix.
    Nested children (e.g. jupyter-sub-subsub) are not included.
    """
    subcommand_tuples = set()
    for d in _path_with_self():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name.startswith('jupyter-'):
                if sys.platform.startswith('win'):
                    name = os.path.splitext(name)[0]
                subcommand_tuples.add(tuple(name.split('-')[1:]))
    subcommands = set()
    for sub_tup in subcommand_tuples:
        if not any((sub_tup[:i] in subcommand_tuples for i in range(1, len(sub_tup)))):
            subcommands.add('-'.join(sub_tup))
    return sorted(subcommands)