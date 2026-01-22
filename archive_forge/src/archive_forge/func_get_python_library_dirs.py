import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import numpy as np
import functools
import os
import subprocess
import sys
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path
def get_python_library_dirs(self):
    """
        Get the library directories necessary to link with Python.
        """
    return list(self._py_lib_dirs)