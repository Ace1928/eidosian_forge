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
def test_check_duplicates(self):
    for i in range(4):
        lines = self.make_tag_lines()
        lines.insert(i, lines[i])
        self.assertCheckFails(Tag, b'\n'.join(lines))