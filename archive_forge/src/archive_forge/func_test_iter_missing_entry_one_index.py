from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_missing_entry_one_index(self):
    idx1 = self.make_index('1')
    idx = _mod_index.CombinedGraphIndex([idx1])
    self.assertEqual([], list(idx.iter_entries([(b'a',)])))