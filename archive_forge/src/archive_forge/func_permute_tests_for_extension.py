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
def permute_tests_for_extension(standard_tests, loader, py_module_name, ext_module_name):
    """Helper for permutating tests against an extension module.

    This is meant to be used inside a modules 'load_tests()' function. It will
    create 2 scenarios, and cause all tests in the 'standard_tests' to be run
    against both implementations. Setting 'test.module' to the appropriate
    module. See breezy.tests.test__chk_map.load_tests as an example.

    :param standard_tests: A test suite to permute
    :param loader: A TestLoader
    :param py_module_name: The python path to a python module that can always
        be loaded, and will be considered the 'python' implementation. (eg
        'breezy._chk_map_py')
    :param ext_module_name: The python path to an extension module. If the
        module cannot be loaded, a single test will be added, which notes that
        the module is not available. If it can be loaded, all standard_tests
        will be run against that module.
    :return: (suite, feature) suite is a test-suite that has all the permuted
        tests. feature is the Feature object that can be used to determine if
        the module is available.
    """
    from .features import ModuleAvailableFeature
    py_module = pyutils.get_named_object(py_module_name)
    scenarios = [('python', {'module': py_module})]
    suite = loader.suiteClass()
    feature = ModuleAvailableFeature(ext_module_name)
    if feature.available():
        scenarios.append(('C', {'module': feature.module}))
    else:

        class FailWithoutFeature(TestCase):

            def id(self):
                return ext_module_name + '.' + super().id()

            def test_fail(self):
                self.requireFeature(feature)
        suite.addTest(loader.loadTestsFromTestCase(FailWithoutFeature))
    result = multiply_tests(standard_tests, scenarios, suite)
    return (result, feature)