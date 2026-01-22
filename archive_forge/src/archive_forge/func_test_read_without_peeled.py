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
def test_read_without_peeled(self):
    f = BytesIO(b'\n'.join([b'# comment', ONES + b' ref/1', TWOS + b' ref/2']))
    self.assertEqual([(ONES, b'ref/1'), (TWOS, b'ref/2')], list(read_packed_refs(f)))