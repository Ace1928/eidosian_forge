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
def test_blob_serialize(self):
    blob = make_object(Blob, data=b'i am a blob')
    with self.assert_serialization_on_change(blob, needs_serialization_after_change=False):
        blob.data = b'i am another blob'