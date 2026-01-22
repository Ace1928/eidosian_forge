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
def test_read_with_peeled_errors(self):
    f = BytesIO(b'\n'.join([b'^' + TWOS, ONES + b' ref/1']))
    self.assertRaises(errors.PackedRefsException, list, read_packed_refs(f))
    f = BytesIO(b'\n'.join([ONES + b' ref/1', b'^' + TWOS, b'^' + THREES]))
    self.assertRaises(errors.PackedRefsException, list, read_packed_refs(f))