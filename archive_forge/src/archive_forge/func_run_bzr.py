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
def run_bzr(self, args, retcode=0, stdin=None, encoding=None, working_dir=None, error_regexes=[]):
    """Invoke brz, as if it were run from the command line.

        The argument list should not include the brz program name - the
        first argument is normally the brz command.  Arguments may be
        passed in three ways:

        1- A list of strings, eg ["commit", "a"].  This is recommended
        when the command contains whitespace or metacharacters, or
        is built up at run time.

        2- A single string, eg "add a".  This is the most convenient
        for hardcoded commands.

        This runs brz through the interface that catches and reports
        errors, and with logging set to something approximating the
        default, so that error reporting can be checked.

        This should be the main method for tests that want to exercise the
        overall behavior of the brz application (rather than a unit test
        or a functional test of the library.)

        This sends the stdout/stderr results into the test's log,
        where it may be useful for debugging.  See also run_captured.

        :keyword stdin: A string to be used as stdin for the command.
        :keyword retcode: The status code the command should return;
            default 0.
        :keyword working_dir: The directory to run the command in
        :keyword error_regexes: A list of expected error messages.  If
            specified they must be seen in the error output of the command.
        """
    if isinstance(args, str):
        args = shlex.split(args)
    if encoding is None:
        encoding = osutils.get_user_encoding()
    stdout = ui_testing.StringIOWithEncoding()
    stderr = ui_testing.StringIOWithEncoding()
    stdout.encoding = stderr.encoding = encoding
    handler = logging.StreamHandler(stream=stderr)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger('')
    logger.addHandler(handler)
    try:
        result = self._run_bzr_core(args, encoding=encoding, stdin=stdin, stdout=stdout, stderr=stderr, working_dir=working_dir)
    finally:
        logger.removeHandler(handler)
    out = stdout.getvalue()
    err = stderr.getvalue()
    if out:
        self.log('output:\n%r', out)
    if err:
        self.log('errors:\n%r', err)
    if retcode is not None:
        self.assertEqual(retcode, result, message='Unexpected return code')
    self.assertIsInstance(error_regexes, (list, tuple))
    for regex in error_regexes:
        self.assertContainsRe(err, regex)
    return (out, err)