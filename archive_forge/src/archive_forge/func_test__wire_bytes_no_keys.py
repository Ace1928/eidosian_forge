import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test__wire_bytes_no_keys(self):
    locations, block = self.make_block(self._texts)
    manager = groupcompress._LazyGroupContentManager(block)
    wire_bytes = manager._wire_bytes()
    block_length = len(block.to_bytes())
    stripped_block = manager._block.to_bytes()
    self.assertTrue(block_length > len(stripped_block))
    empty_z_header = zlib.compress(b'')
    self.assertEqual(b'groupcompress-block\n8\n0\n%d\n%s%s' % (len(stripped_block), empty_z_header, stripped_block), wire_bytes)