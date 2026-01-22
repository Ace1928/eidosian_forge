from ... import errors, tests, transport
from .. import index as _mod_index
def test_iteration_absent_skipped(self):
    index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],))])
    self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),))}, set(index.iter_all_entries()))
    self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),))}, set(index.iter_entries([(b'name',)])))
    self.assertEqual([], list(index.iter_entries([(b'ref',)])))