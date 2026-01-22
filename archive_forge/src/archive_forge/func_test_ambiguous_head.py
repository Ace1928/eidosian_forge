from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_ambiguous_head(self):
    r = {b'refs/heads/ambig4': 'bla', b'refs/remotes/ambig4': 'bla', b'refs/remotes/ambig4/HEAD': 'bla'}
    self.assertEqual(b'refs/heads/ambig4', parse_ref(r, b'ambig4'))