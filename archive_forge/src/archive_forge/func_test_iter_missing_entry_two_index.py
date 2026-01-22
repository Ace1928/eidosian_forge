from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_missing_entry_two_index(self):
    idx1 = self.make_index('1')
    idx2 = self.make_index('2')
    idx = _mod_index.CombinedGraphIndex([idx1, idx2])
    self.assertEqual([], list(idx.iter_entries([('a',)])))