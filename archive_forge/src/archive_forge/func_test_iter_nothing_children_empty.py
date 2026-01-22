from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_nothing_children_empty(self):
    idx1 = self.make_index('name')
    idx = _mod_index.CombinedGraphIndex([idx1])
    self.assertEqual([], list(idx.iter_entries([])))