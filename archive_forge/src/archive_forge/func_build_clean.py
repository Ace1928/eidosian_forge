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
def build_clean(self) -> int:
    srcdir = path.abspath(self.srcdir)
    builddir = path.abspath(self.builddir)
    if not path.exists(self.builddir):
        return 0
    elif not path.isdir(self.builddir):
        print('Error: %r is not a directory!' % self.builddir)
        return 1
    elif srcdir == builddir:
        print('Error: %r is same as source directory!' % self.builddir)
        return 1
    elif path.commonpath([srcdir, builddir]) == builddir:
        print('Error: %r directory contains source directory!' % self.builddir)
        return 1
    print('Removing everything under %r...' % self.builddir)
    for item in os.listdir(self.builddir):
        rmtree(self.builddir_join(item))
    return 0