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
def knownFailure(self, reason):
    """Declare that this test fails for a known reason

        Tests that are known to fail should generally be using expectedFailure
        with an appropriate reverse assertion if a change could cause the test
        to start passing. Conversely if the test has no immediate prospect of
        succeeding then using skip is more suitable.

        When this method is called while an exception is being handled, that
        traceback will be used, otherwise a new exception will be thrown to
        provide one but won't be reported.
        """
    self._add_reason(reason)
    try:
        exc_info = sys.exc_info()
        if exc_info != (None, None, None):
            self._report_traceback(exc_info)
        else:
            try:
                raise self.failureException(reason)
            except self.failureException:
                exc_info = sys.exc_info()
        raise testtools.testcase._ExpectedFailure(exc_info)
    finally:
        del exc_info