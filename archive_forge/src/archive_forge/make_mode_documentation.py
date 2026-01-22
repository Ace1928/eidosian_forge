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
sphinx-build -M command-line handling.

This replaces the old, platform-dependent and once-generated content
of Makefile / make.bat.

This is in its own module so that importing it is fast.  It should not
import the main Sphinx modules (like sphinx.applications, sphinx.builders).
