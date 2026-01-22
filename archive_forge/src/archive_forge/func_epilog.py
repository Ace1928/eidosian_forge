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
@epilog.setter
def epilog(self, x: Any) -> None:
    """Ignore epilog set in Parser.__init__"""