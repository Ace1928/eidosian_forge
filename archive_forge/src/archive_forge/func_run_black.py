from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
import contextlib
import difflib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union
from . import compat
def run_black(self, tempfile: str) -> None:
    self._run_console_script(str(tempfile), {'entrypoint': 'black', 'options': f'--config {self.pyproject_toml_path}'})