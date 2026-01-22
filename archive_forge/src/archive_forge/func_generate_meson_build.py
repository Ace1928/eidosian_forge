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
def generate_meson_build(self):
    for node in self.pipeline:
        node()
    template = Template(self.meson_build_template())
    return template.substitute(self.substitutions)