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
def test_tag_serialize_time_error(self):
    with self.assertRaises(ObjectFormatException):
        tag = make_object(Tag, name=b'tag', message=b'some message', tagger=b'Tagger <test@example.com> 1174773719+0000', object=(Commit, b'0' * 40))
        tag._deserialize(tag._serialize())