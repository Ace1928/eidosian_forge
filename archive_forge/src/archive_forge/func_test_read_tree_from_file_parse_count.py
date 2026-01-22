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
def test_read_tree_from_file_parse_count(self):
    old_deserialize = Tree._deserialize

    def reset_deserialize():
        Tree._deserialize = old_deserialize
    self.addCleanup(reset_deserialize)
    self.deserialize_count = 0

    def counting_deserialize(*args, **kwargs):
        self.deserialize_count += 1
        return old_deserialize(*args, **kwargs)
    Tree._deserialize = counting_deserialize
    t = self.get_tree(tree_sha)
    self.assertEqual(t.items()[0], (b'a', 33188, a_sha))
    self.assertEqual(t.items()[1], (b'b', 33188, b_sha))
    self.assertEqual(self.deserialize_count, 1)