import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_yield_factories_calls_get_blocks(self):
    """Uncached memos are retrieved via get_blocks."""
    read_memo1 = ('fake index', 100, 50)
    read_memo2 = ('fake index', 150, 40)
    gcvf = StubGCVF(canned_get_blocks=[(read_memo1, groupcompress.GroupCompressBlock()), (read_memo2, groupcompress.GroupCompressBlock())])
    locations = {('key1',): (read_memo1 + (0, 0), None, None, None), ('key2',): (read_memo2 + (0, 0), None, None, None)}
    batcher = groupcompress._BatchingBlockFetcher(gcvf, locations)
    batcher.add_key(('key1',))
    batcher.add_key(('key2',))
    factories = list(batcher.yield_factories(full_flush=True))
    self.assertLength(2, factories)
    keys = [f.key for f in factories]
    kinds = [f.storage_kind for f in factories]
    self.assertEqual([('key1',), ('key2',)], keys)
    self.assertEqual(['groupcompress-block', 'groupcompress-block'], kinds)