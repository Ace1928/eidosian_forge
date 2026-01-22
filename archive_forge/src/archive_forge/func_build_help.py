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
def build_help(self) -> None:
    if not color_terminal():
        nocolor()
    print(bold('Sphinx v%s' % sphinx.__display_version__))
    print("Please use `make %s' where %s is one of" % ((blue('target'),) * 2))
    for osname, bname, description in BUILDERS:
        if not osname or os.name == osname:
            print('  %s  %s' % (blue(bname.ljust(10)), description))