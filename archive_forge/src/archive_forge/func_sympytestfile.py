import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def sympytestfile(filename, module_relative=True, name=None, package=None, globs=None, verbose=None, report=True, optionflags=0, extraglobs=None, raise_on_error=False, parser=pdoctest.DocTestParser(), encoding=None):
    """
    Test examples in the given file.  Return (#failures, #tests).

    Optional keyword arg ``module_relative`` specifies how filenames
    should be interpreted:

    - If ``module_relative`` is True (the default), then ``filename``
      specifies a module-relative path.  By default, this path is
      relative to the calling module's directory; but if the
      ``package`` argument is specified, then it is relative to that
      package.  To ensure os-independence, ``filename`` should use
      "/" characters to separate path segments, and should not
      be an absolute path (i.e., it may not begin with "/").

    - If ``module_relative`` is False, then ``filename`` specifies an
      os-specific path.  The path may be absolute or relative (to
      the current working directory).

    Optional keyword arg ``name`` gives the name of the test; by default
    use the file's basename.

    Optional keyword argument ``package`` is a Python package or the
    name of a Python package whose directory should be used as the
    base directory for a module relative filename.  If no package is
    specified, then the calling module's directory is used as the base
    directory for module relative filenames.  It is an error to
    specify ``package`` if ``module_relative`` is False.

    Optional keyword arg ``globs`` gives a dict to be used as the globals
    when executing examples; by default, use {}.  A copy of this dict
    is actually used for each docstring, so that each docstring's
    examples start with a clean slate.

    Optional keyword arg ``extraglobs`` gives a dictionary that should be
    merged into the globals that are used to execute examples.  By
    default, no extra globals are used.

    Optional keyword arg ``verbose`` prints lots of stuff if true, prints
    only failures if false; by default, it's true iff "-v" is in sys.argv.

    Optional keyword arg ``report`` prints a summary at the end when true,
    else prints nothing at the end.  In verbose mode, the summary is
    detailed, else very brief (in fact, empty if all tests passed).

    Optional keyword arg ``optionflags`` or's together module constants,
    and defaults to 0.  Possible values (see the docs for details):

    - DONT_ACCEPT_TRUE_FOR_1
    - DONT_ACCEPT_BLANKLINE
    - NORMALIZE_WHITESPACE
    - ELLIPSIS
    - SKIP
    - IGNORE_EXCEPTION_DETAIL
    - REPORT_UDIFF
    - REPORT_CDIFF
    - REPORT_NDIFF
    - REPORT_ONLY_FIRST_FAILURE

    Optional keyword arg ``raise_on_error`` raises an exception on the
    first unexpected exception or failure. This allows failures to be
    post-mortem debugged.

    Optional keyword arg ``parser`` specifies a DocTestParser (or
    subclass) that should be used to extract tests from the files.

    Optional keyword arg ``encoding`` specifies an encoding that should
    be used to convert the file to unicode.

    Advanced tomfoolery:  testmod runs methods of a local instance of
    class doctest.Tester, then merges the results into (or creates)
    global Tester instance doctest.master.  Methods of doctest.master
    can be called directly too, if you want to do something unusual.
    Passing report=0 to testmod is especially useful then, to delay
    displaying a summary.  Invoke doctest.master.summarize(verbose)
    when you're done fiddling.
    """
    if package and (not module_relative):
        raise ValueError('Package may only be specified for module-relative paths.')
    text, filename = pdoctest._load_testfile(filename, package, module_relative, encoding)
    if name is None:
        name = os.path.basename(filename)
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    if extraglobs is not None:
        globs.update(extraglobs)
    if '__name__' not in globs:
        globs['__name__'] = '__main__'
    if raise_on_error:
        runner = pdoctest.DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = SymPyDocTestRunner(verbose=verbose, optionflags=optionflags)
        runner._checker = SymPyOutputChecker()
    test = parser.get_doctest(text, globs, name, filename, 0)
    runner.run(test)
    if report:
        runner.summarize()
    if pdoctest.master is None:
        pdoctest.master = runner
    else:
        pdoctest.master.merge(runner)
    return SymPyTestResults(runner.failures, runner.tries)