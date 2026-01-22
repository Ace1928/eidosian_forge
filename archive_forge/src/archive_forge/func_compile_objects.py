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
def compile_objects(self, sources, output_dir, include_dirs=(), depends=(), macros=(), extra_cflags=None):
    """
        Compile the given source files into a separate object file each,
        all beneath the *output_dir*.  A list of paths to object files
        is returned.

        *macros* has the same format as in distutils: a list of 1- or 2-tuples.
        If a 1-tuple (name,), the given name is considered undefined by
        the C preprocessor.
        If a 2-tuple (name, value), the given name is expanded into the
        given value by the C preprocessor.
        """
    objects = self._compiler.compile(sources, output_dir=output_dir, include_dirs=include_dirs, depends=depends, macros=macros or [], extra_preargs=extra_cflags)
    return objects