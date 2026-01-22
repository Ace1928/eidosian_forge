import argparse
import bdb
import locale
import multiprocessing
import os
import pdb
import sys
import traceback
from os import path
from typing import Any, List, Optional, TextIO
from docutils.utils import SystemMessage
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import Tee, format_exception_cut_frames, save_traceback
from sphinx.util.console import color_terminal, nocolor, red, terminal_safe  # type: ignore
from sphinx.util.docutils import docutils_namespace, patch_docutils
from sphinx.util.osutil import abspath, ensuredir
def make_main(argv: List[str]=sys.argv[1:]) -> int:
    """Sphinx build "make mode" entry."""
    from sphinx.cmd import make_mode
    return make_mode.run_make_mode(argv[1:])