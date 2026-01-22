from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_ambiguous_ref(self):
    r = {b'ambig1': 'bla', b'refs/ambig1': 'bla', b'refs/tags/ambig1': 'bla', b'refs/heads/ambig1': 'bla', b'refs/remotes/ambig1': 'bla', b'refs/remotes/ambig1/HEAD': 'bla'}
    self.assertEqual(b'ambig1', parse_ref(r, b'ambig1'))