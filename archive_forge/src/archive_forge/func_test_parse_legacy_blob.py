import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_parse_legacy_blob(self):
    string = b'test 3\n'
    b = self.get_blob(c_sha)
    self.assertEqual(b.data, string)
    self.assertEqual(b.sha().hexdigest().encode('ascii'), c_sha)