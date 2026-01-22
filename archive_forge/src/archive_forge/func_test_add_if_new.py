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
def test_add_if_new(self):
    nines = b'9' * 40
    self.assertFalse(self._refs.add_if_new(b'refs/heads/master', nines))
    self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs[b'refs/heads/master'])
    self.assertTrue(self._refs.add_if_new(b'refs/some/ref', nines))
    self.assertEqual(nines, self._refs[b'refs/some/ref'])