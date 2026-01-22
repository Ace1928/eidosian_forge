import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def run_suite(suite, name='test', verbose=False, pattern='.*', stop_on_failure=False, transport=None, lsprof_timed=None, bench_history=None, matching_tests_first=None, list_only=False, random_seed=None, exclude_pattern=None, strict=False, runner_class=None, suite_decorators=None, stream=None, result_decorators=None):
    """Run a test suite for brz selftest.

    :param runner_class: The class of runner to use. Must support the
        constructor arguments passed by run_suite which are more than standard
        python uses.
    :return: A boolean indicating success.
    """
    TestCase._gather_lsprof_in_benchmarks = lsprof_timed
    if verbose:
        verbosity = 2
    else:
        verbosity = 1
    if runner_class is None:
        runner_class = TextTestRunner
    if stream is None:
        stream = sys.stdout
    runner = runner_class(stream=stream, descriptions=0, verbosity=verbosity, bench_history=bench_history, strict=strict, result_decorators=result_decorators)
    runner.stop_on_failure = stop_on_failure
    if isinstance(suite, unittest.TestSuite):
        suite._tests[:], suite = ([], TestSuite(suite))
    decorators = [random_order(random_seed, runner), exclude_tests(exclude_pattern)]
    if matching_tests_first:
        decorators.append(tests_first(pattern))
    else:
        decorators.append(filter_tests(pattern))
    if suite_decorators:
        decorators.extend(suite_decorators)
    if fork_decorator not in decorators:
        decorators.append(CountingDecorator)
    for decorator in decorators:
        suite = decorator(suite)
    if list_only:
        if verbosity >= 2:
            stream.write('Listing tests only ...\n')
        if getattr(runner, 'list', None) is not None:
            runner.list(suite)
        else:
            for t in iter_suite_tests(suite):
                stream.write('%s\n' % t.id())
        return True
    result = runner.run(suite)
    if strict and getattr(result, 'wasStrictlySuccessful', False):
        return result.wasStrictlySuccessful()
    else:
        return result.wasSuccessful()