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
def test_read_loose_ref(self):
    self._refs[b'refs/heads/foo'] = b'df6800012397fb85c56e7418dd4eb9405dee075c'
    self.assertEqual(None, self._refs.read_ref(b'refs/heads/foo/bar'))