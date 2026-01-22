from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_ambiguous_tag(self):
    r = {b'refs/tags/ambig3': 'bla', b'refs/heads/ambig3': 'bla', b'refs/remotes/ambig3': 'bla', b'refs/remotes/ambig3/HEAD': 'bla'}
    self.assertEqual(b'refs/tags/ambig3', parse_ref(r, b'ambig3'))