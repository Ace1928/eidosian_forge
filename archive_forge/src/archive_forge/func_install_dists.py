import os
import operator
import sys
import contextlib
import itertools
import unittest
from distutils.errors import DistutilsError, DistutilsOptionError
from distutils import log
from unittest import TestLoader
from pkg_resources import (
from .._importlib import metadata
from setuptools import Command
from setuptools.extern.more_itertools import unique_everseen
from setuptools.extern.jaraco.functools import pass_none
@staticmethod
def install_dists(dist):
    """
        Install the requirements indicated by self.distribution and
        return an iterable of the dists that were built.
        """
    ir_d = dist.fetch_build_eggs(dist.install_requires)
    tr_d = dist.fetch_build_eggs(dist.tests_require or [])
    er_d = dist.fetch_build_eggs((v for k, v in dist.extras_require.items() if k.startswith(':') and evaluate_marker(k[1:])))
    return itertools.chain(ir_d, tr_d, er_d)