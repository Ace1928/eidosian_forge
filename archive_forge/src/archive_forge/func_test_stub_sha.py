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
def test_stub_sha(self):
    sha = b'5' * 40
    c = make_commit(id=sha, message=b'foo')
    self.assertIsInstance(c, Commit)
    self.assertEqual(sha, c.id)
    self.assertNotEqual(sha, c.sha())