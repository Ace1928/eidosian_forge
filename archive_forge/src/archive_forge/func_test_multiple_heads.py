from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_multiple_heads(self):
    self.assertSearchResult([b'e', b'f'], [b'a'], 5, extended_history_shortcut, (), [b'a'], 10)
    self.assertSearchResult([b'f'], [b'a'], 4, extended_history_shortcut, (), [b'a'], 1)
    self.assertSearchResult([b'f'], [b'a'], 4, extended_history_shortcut, (), [b'a'], 2)