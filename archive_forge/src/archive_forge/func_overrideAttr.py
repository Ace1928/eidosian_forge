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
def overrideAttr(self, obj, attr_name, new=_unitialized_attr):
    """Overrides an object attribute restoring it after the test.

        :note: This should be used with discretion; you should think about
        whether it's better to make the code testable without monkey-patching.

        :param obj: The object that will be mutated.

        :param attr_name: The attribute name we want to preserve/override in
            the object.

        :param new: The optional value we want to set the attribute to.

        :returns: The actual attr value.
        """
    value = getattr(obj, attr_name, _unitialized_attr)
    if value is _unitialized_attr:
        if new is not _unitialized_attr:
            self.addCleanup(delattr, obj, attr_name)
    else:
        self.addCleanup(setattr, obj, attr_name, value)
    if new is not _unitialized_attr:
        setattr(obj, attr_name, new)
    return value