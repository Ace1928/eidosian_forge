from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_nodes_no_refs(self):
    index = self.make_index(0)
    index.add_nodes([((b'name',), b'data')])
    index.add_nodes([((b'name2',), b''), ((b'name3',), b'')])
    self.assertEqual({(index, (b'name',), b'data'), (index, (b'name2',), b''), (index, (b'name3',), b'')}, set(index.iter_all_entries()))