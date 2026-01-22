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
def test_invalid_refname(self):
    text = _TEST_REFS_SERIALIZED + b'00' * 20 + b'\trefs/stash\n'
    refs = InfoRefsContainer(BytesIO(text))
    expected_refs = dict(_TEST_REFS)
    del expected_refs[b'HEAD']
    expected_refs[b'refs/stash'] = b'00' * 20
    del expected_refs[b'refs/heads/loop']
    self.assertEqual(expected_refs, refs.as_dict())