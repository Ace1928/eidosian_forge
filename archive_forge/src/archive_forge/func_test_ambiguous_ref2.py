from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_ambiguous_ref2(self):
    r = {b'refs/ambig2': 'bla', b'refs/tags/ambig2': 'bla', b'refs/heads/ambig2': 'bla', b'refs/remotes/ambig2': 'bla', b'refs/remotes/ambig2/HEAD': 'bla'}
    self.assertEqual(b'refs/ambig2', parse_ref(r, b'ambig2'))