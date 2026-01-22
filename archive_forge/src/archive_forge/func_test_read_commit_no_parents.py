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
def test_read_commit_no_parents(self):
    sha = b'0d89f20333fbb1d2f3a94da77f4981373d8f4310'
    c = self.commit(sha)
    self.assertEqual(c.tree, b'90182552c4a85a45ec2a835cadc3451bebdfe870')
    self.assertEqual(c.parents, [])
    self.assertEqual(c.author, b'James Westby <jw+debian@jameswestby.net>')
    self.assertEqual(c.committer, b'James Westby <jw+debian@jameswestby.net>')
    self.assertEqual(c.commit_time, 1174758034)
    self.assertEqual(c.commit_timezone, 0)
    self.assertEqual(c.author_timezone, 0)
    self.assertEqual(c.message, b'Test commit\n')