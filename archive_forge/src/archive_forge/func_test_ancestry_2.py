from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_ancestry_2(self):
    self.assertSearchResult([b'rev1b', b'rev4a'], [NULL_REVISION], len(ancestry_2), ancestry_2)
    self.assertSearchResult([b'rev1b', b'rev4a'], [], len(ancestry_2) + 1, ancestry_2, missing_keys=[NULL_REVISION])