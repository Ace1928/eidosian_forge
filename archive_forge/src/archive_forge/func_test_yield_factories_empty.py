import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_yield_factories_empty(self):
    """An empty batch yields no factories."""
    batcher = groupcompress._BatchingBlockFetcher(StubGCVF(), {})
    self.assertEqual([], list(batcher.yield_factories()))