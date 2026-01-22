import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
def test_write_without_peeled(self):
    f = BytesIO()
    write_packed_refs(f, {b'ref/1': ONES, b'ref/2': TWOS})
    self.assertEqual(b'\n'.join([ONES + b' ref/1', TWOS + b' ref/2']) + b'\n', f.getvalue())