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
def test_format_timezone_pdt_half(self):
    self.assertEqual(b'-0440', format_timezone(int((-4 * 60 - 40) * 60)))