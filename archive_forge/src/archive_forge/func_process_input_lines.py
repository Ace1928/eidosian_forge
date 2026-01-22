import atexit
import errno
import os
import pathlib
import re
import sys
import tempfile
import ast
import warnings
import shutil
from io import StringIO
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive
from sphinx.util import logging
from traitlets.config import Config
from IPython import InteractiveShell
from IPython.core.profiledir import ProfileDir
def process_input_lines(self, lines, store_history=True):
    """process the input, capturing stdout"""
    stdout = sys.stdout
    source_raw = '\n'.join(lines)
    try:
        sys.stdout = self.cout
        self.IP.run_cell(source_raw, store_history=store_history)
    finally:
        sys.stdout = stdout