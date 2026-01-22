from ... import errors, tests, transport
from .. import index as _mod_index
def test_key_count_sums_index_keys(self):
    idx1 = self.make_index('1', nodes=[((b'1',), b'', ()), ((b'2',), b'', ())])
    idx2 = self.make_index('2', nodes=[((b'1',), b'', ())])
    idx = _mod_index.CombinedGraphIndex([idx1, idx2])
    self.assertEqual(3, idx.key_count())