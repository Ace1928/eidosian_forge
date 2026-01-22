import the main Sphinx modules (like sphinx.applications, sphinx.builders).
import os
import subprocess
import sys
from os import path
from typing import List, Optional
import sphinx
from sphinx.cmd.build import build_main
from sphinx.util.console import blue, bold, color_terminal, nocolor  # type: ignore
from sphinx.util.osutil import cd, rmtree
def builddir_join(self, *comps: str) -> str:
    return path.join(self.builddir, *comps)