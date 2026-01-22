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
def run_make_mode(args: List[str]) -> int:
    if len(args) < 3:
        print('Error: at least 3 arguments (builder, source dir, build dir) are required.', file=sys.stderr)
        return 1
    make = Make(args[1], args[2], args[3:])
    run_method = 'build_' + args[0]
    if hasattr(make, run_method):
        return getattr(make, run_method)()
    return make.run_generic_build(args[0])