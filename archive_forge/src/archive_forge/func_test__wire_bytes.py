import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test__wire_bytes(self):
    locations, block = self.make_block(self._texts)
    manager = groupcompress._LazyGroupContentManager(block)
    self.add_key_to_manager((b'key1',), locations, block, manager)
    self.add_key_to_manager((b'key4',), locations, block, manager)
    block_bytes = block.to_bytes()
    wire_bytes = manager._wire_bytes()
    storage_kind, z_header_len, header_len, block_len, rest = wire_bytes.split(b'\n', 4)
    z_header_len = int(z_header_len)
    header_len = int(header_len)
    block_len = int(block_len)
    self.assertEqual(b'groupcompress-block', storage_kind)
    self.assertEqual(34, z_header_len)
    self.assertEqual(26, header_len)
    self.assertEqual(len(block_bytes), block_len)
    z_header = rest[:z_header_len]
    header = zlib.decompress(z_header)
    self.assertEqual(header_len, len(header))
    entry1 = locations[b'key1',]
    entry4 = locations[b'key4',]
    self.assertEqualDiff(b'key1\n\n%d\n%d\nkey4\n\n%d\n%d\n' % (entry1[0], entry1[1], entry4[0], entry4[1]), header)
    z_block = rest[z_header_len:]
    self.assertEqual(block_bytes, z_block)