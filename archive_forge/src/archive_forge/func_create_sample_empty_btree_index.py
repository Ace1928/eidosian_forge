from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def create_sample_empty_btree_index(self):
    builder = btree_index.BTreeBuilder(reference_lists=1, key_elements=2)
    out_f = builder.finish()
    try:
        self.build_tree_contents([('test.btree', out_f.read())])
    finally:
        out_f.close()