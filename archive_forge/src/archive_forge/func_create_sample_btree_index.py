from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def create_sample_btree_index(self):
    builder = btree_index.BTreeBuilder(reference_lists=1, key_elements=2)
    builder.add_node((b'test', b'key1'), b'value', (((b'ref', b'entry'),),))
    builder.add_node((b'test', b'key2'), b'value2', (((b'ref', b'entry2'),),))
    builder.add_node((b'test2', b'key3'), b'value3', (((b'ref', b'entry3'),),))
    out_f = builder.finish()
    try:
        self.build_tree_contents([('test.btree', out_f.read())])
    finally:
        out_f.close()