import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_manager_custom_compressor_settings(self):
    locations, old_block = self.make_block(self._texts)
    called = []

    def compressor_settings():
        called.append('called')
        return (10,)
    manager = groupcompress._LazyGroupContentManager(old_block, get_compressor_settings=compressor_settings)
    gcvf = groupcompress.GroupCompressVersionedFiles
    self.assertIs(None, manager._compressor_settings)
    self.assertEqual((10,), manager._get_compressor_settings())
    self.assertEqual((10,), manager._get_compressor_settings())
    self.assertEqual((10,), manager._compressor_settings)
    self.assertEqual(['called'], called)