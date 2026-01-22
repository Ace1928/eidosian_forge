import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def clone_test_with_new_id(test, new_id):
    """Copy a `TestCase`, and give the copied test a new id.

    This is only expected to be used on tests that have been constructed but
    not executed.
    """
    return _clone_test_id_callback(test, lambda: new_id)