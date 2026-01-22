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
def test_check_tag_with_overflow_time(self):
    """Date with overflow should raise an ObjectFormatException when checked."""
    author = f'Some Dude <some@dude.org> {MAX_TIME + 1} +0000'
    tag = Tag.from_string(self.make_tag_text(tagger=author.encode()))
    with self.assertRaises(ObjectFormatException):
        tag.check()