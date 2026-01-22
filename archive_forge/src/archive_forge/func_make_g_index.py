import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def make_g_index(self, name, ref_lists=0, nodes=[]):
    builder = btree_index.BTreeBuilder(ref_lists)
    for node, references, value in nodes:
        builder.add_node(node, references, value)
    stream = builder.finish()
    trans = self.get_transport()
    size = trans.put_file(name, stream)
    return btree_index.BTreeGraphIndex(trans, name, size)