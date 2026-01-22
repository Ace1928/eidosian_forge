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
def custom_doctest(self, decorator, input_lines, found, submitted):
    """
        Perform a specialized doctest.

        """
    from .custom_doctests import doctests
    args = decorator.split()
    doctest_type = args[1]
    if doctest_type in doctests:
        doctests[doctest_type](self, args, input_lines, found, submitted)
    else:
        e = 'Invalid option to @doctest: {0}'.format(doctest_type)
        raise Exception(e)