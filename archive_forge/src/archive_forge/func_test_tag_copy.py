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
def test_tag_copy(self):
    tag = make_object(Tag, name=b'tag', message=b'', tagger=b'Tagger <test@example.com>', tag_time=12345, tag_timezone=0, object=(Commit, b'0' * 40))
    self.assert_copy(tag)