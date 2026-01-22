from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def test_dump_btree_no_refs_smoke(self):
    builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=2)
    builder.add_node((b'test', b'key1'), b'value')
    out_f = builder.finish()
    try:
        self.build_tree_contents([('test.btree', out_f.read())])
    finally:
        out_f.close()
    out, err = self.run_bzr('dump-btree test.btree')