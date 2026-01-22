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
def build_gettext(self) -> int:
    dtdir = self.builddir_join('gettext', '.doctrees')
    if self.run_generic_build('gettext', doctreedir=dtdir) > 0:
        return 1
    return 0