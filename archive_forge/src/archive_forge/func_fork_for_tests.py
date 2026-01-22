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
def fork_for_tests(suite):
    """Take suite and start up one runner per CPU by forking()

    :return: An iterable of TestCase-like objects which can each have
        run(result) called on them to feed tests to result.
    """
    concurrency = osutils.local_concurrency()
    result = []
    from subunit import ProtocolTestCase
    from subunit.test_results import AutoTimingTestResultDecorator

    class TestInOtherProcess(ProtocolTestCase):

        def __init__(self, stream, pid):
            ProtocolTestCase.__init__(self, stream)
            self.pid = pid

        def run(self, result):
            try:
                ProtocolTestCase.run(self, result)
            finally:
                pid, status = os.waitpid(self.pid, 0)
    test_blocks = partition_tests(suite, concurrency)
    suite._tests[:] = []
    for process_tests in test_blocks:
        process_suite = TestUtil.TestSuite(process_tests)
        process_tests[:] = []
        c2pread, c2pwrite = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                stream = os.fdopen(c2pwrite, 'wb', 0)
                workaround_zealous_crypto_random()
                try:
                    import coverage
                except ModuleNotFoundError:
                    pass
                else:
                    coverage.process_startup()
                os.close(c2pread)
                sys.stdin.close()
                subunit_result = AutoTimingTestResultDecorator(SubUnitBzrProtocolClientv1(stream))
                process_suite.run(subunit_result)
            except:
                tb = traceback.format_exc()
                if isinstance(tb, str):
                    tb = tb.encode('utf-8')
                try:
                    stream.write(tb)
                finally:
                    stream.flush()
                    os._exit(1)
            os._exit(0)
        else:
            os.close(c2pwrite)
            stream = os.fdopen(c2pread, 'rb', 0)
            test = TestInOtherProcess(stream, pid)
            result.append(test)
    return result