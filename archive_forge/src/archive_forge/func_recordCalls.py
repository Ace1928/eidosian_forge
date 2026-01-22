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
def recordCalls(self, obj, attr_name):
    """Monkeypatch in a wrapper that will record calls.

        The monkeypatch is automatically removed when the test concludes.

        :param obj: The namespace holding the reference to be replaced;
            typically a module, class, or object.
        :param attr_name: A string for the name of the attribute to patch.
        :returns: A list that will be extended with one item every time the
            function is called, with a tuple of (args, kwargs).
        """
    calls = []

    def decorator(*args, **kwargs):
        calls.append((args, kwargs))
        return orig(*args, **kwargs)
    orig = self.overrideAttr(obj, attr_name, decorator)
    return calls