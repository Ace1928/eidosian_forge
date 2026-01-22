import os
import sys
import platform
import re
import gc
import operator
import warnings
from functools import partial, wraps
import shutil
import contextlib
from tempfile import mkdtemp, mkstemp
from unittest.case import SkipTest
from warnings import WarningMessage
import pprint
import sysconfig
import numpy as np
from numpy.core import (
from numpy import isfinite, isnan, isinf
import numpy.linalg._umath_linalg
from io import StringIO
import unittest
def rundocs(filename=None, raise_on_error=True):
    """
    Run doctests found in the given file.

    By default `rundocs` raises an AssertionError on failure.

    Parameters
    ----------
    filename : str
        The path to the file for which the doctests are run.
    raise_on_error : bool
        Whether to raise an AssertionError when a doctest fails. Default is
        True.

    Notes
    -----
    The doctests can be run by the user/developer by adding the ``doctests``
    argument to the ``test()`` call. For example, to run all tests (including
    doctests) for `numpy.lib`:

    >>> np.lib.test(doctests=True)  # doctest: +SKIP
    """
    from numpy.distutils.misc_util import exec_mod_from_location
    import doctest
    if filename is None:
        f = sys._getframe(1)
        filename = f.f_globals['__file__']
    name = os.path.splitext(os.path.basename(filename))[0]
    m = exec_mod_from_location(name, filename)
    tests = doctest.DocTestFinder().find(m)
    runner = doctest.DocTestRunner(verbose=False)
    msg = []
    if raise_on_error:
        out = lambda s: msg.append(s)
    else:
        out = None
    for test in tests:
        runner.run(test, out=out)
    if runner.failures > 0 and raise_on_error:
        raise AssertionError('Some doctests failed:\n%s' % '\n'.join(msg))