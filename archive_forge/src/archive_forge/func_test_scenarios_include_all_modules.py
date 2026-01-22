import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_scenarios_include_all_modules(self):
    from ..transport import _get_transport_modules
    from .per_transport import transport_test_permutations
    modules = _get_transport_modules()
    permutation_count = 0
    for module in modules:
        try:
            permutation_count += len(reduce(getattr, (module + '.get_test_permutations').split('.')[1:], __import__(module))())
        except errors.DependencyNotPresent:
            pass
    scenarios = transport_test_permutations()
    self.assertEqual(permutation_count, len(scenarios))