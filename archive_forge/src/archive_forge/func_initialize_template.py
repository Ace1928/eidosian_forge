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
def initialize_template(self) -> None:
    self.substitutions['modulename'] = self.modulename
    self.substitutions['buildtype'] = self.build_type
    self.substitutions['python'] = self.python_exe