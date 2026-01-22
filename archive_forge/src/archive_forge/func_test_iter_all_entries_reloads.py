from ... import errors, tests, transport
from .. import index as _mod_index
def test_iter_all_entries_reloads(self):
    index, reload_counter = self.make_combined_index_with_missing()
    result = list(index.iter_all_entries())
    index3 = index._indices[0]
    self.assertEqual({(index3, (b'1',), b''), (index3, (b'2',), b'')}, set(result))
    self.assertEqual([1, 1, 0], reload_counter)