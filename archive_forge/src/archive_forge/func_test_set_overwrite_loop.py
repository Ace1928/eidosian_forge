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
def test_set_overwrite_loop(self):
    self.assertRaises(SymrefLoop, self._refs.follow, b'refs/heads/loop')
    self._refs[b'refs/heads/loop'] = b'42d06bd4b77fed026b154d16493e5deab78f02ec'
    self.assertEqual(([b'refs/heads/loop'], b'42d06bd4b77fed026b154d16493e5deab78f02ec'), self._refs.follow(b'refs/heads/loop'))