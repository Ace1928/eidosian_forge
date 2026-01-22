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
def test_split_ref_line_errors(self):
    self.assertRaises(errors.PackedRefsException, _split_ref_line, b'singlefield')
    self.assertRaises(errors.PackedRefsException, _split_ref_line, b'badsha name')
    self.assertRaises(errors.PackedRefsException, _split_ref_line, ONES + b' bad/../refname')