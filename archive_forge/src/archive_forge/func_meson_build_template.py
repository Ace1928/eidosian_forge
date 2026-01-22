from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
def meson_build_template(self) -> str:
    if not self.build_template_path.is_file():
        raise FileNotFoundError(errno.ENOENT, f'Meson build template {self.build_template_path.absolute()} does not exist.')
    return self.build_template_path.read_text()