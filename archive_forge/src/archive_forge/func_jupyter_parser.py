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
def jupyter_parser() -> JupyterParser:
    """Create a jupyter parser object."""
    parser = JupyterParser(description='Jupyter: Interactive Computing')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--version', action='store_true', help='show the versions of core jupyter packages and exit')
    subcommand_action = group.add_argument('subcommand', type=str, nargs='?', help='the subcommand to launch')
    subcommand_action.completer = lambda *args, **kwargs: list_subcommands()
    group.add_argument('--config-dir', action='store_true', help='show Jupyter config dir')
    group.add_argument('--data-dir', action='store_true', help='show Jupyter data dir')
    group.add_argument('--runtime-dir', action='store_true', help='show Jupyter runtime dir')
    group.add_argument('--paths', action='store_true', help='show all Jupyter paths. Add --json for machine-readable format.')
    parser.add_argument('--json', action='store_true', help='output paths as machine-readable json')
    parser.add_argument('--debug', action='store_true', help='output debug information about paths')
    return parser