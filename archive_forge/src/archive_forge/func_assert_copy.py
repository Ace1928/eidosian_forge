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
def assert_copy(self, orig):
    oclass = object_class(orig.type_num)
    copy = orig.copy()
    self.assertIsInstance(copy, oclass)
    self.assertEqual(copy, orig)
    self.assertIsNot(copy, orig)