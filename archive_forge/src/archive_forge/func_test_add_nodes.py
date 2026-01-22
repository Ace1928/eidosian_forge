from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_nodes(self):
    index, adapter = self.make_index(add_callback=True)
    adapter.add_nodes((((b'key',), b'value', (((b'ref',),),)), ((b'key2',), b'value2', ((),))))
    self.assertEqual({(index, (b'prefix', b'key2'), b'value2', ((),)), (index, (b'prefix', b'key'), b'value', (((b'prefix', b'ref'),),))}, set(index.iter_all_entries()))