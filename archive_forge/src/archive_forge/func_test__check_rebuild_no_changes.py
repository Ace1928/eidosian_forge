import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test__check_rebuild_no_changes(self):
    block, manager = self.make_block_and_full_manager(self._texts)
    manager._check_rebuild_block()
    self.assertIs(block, manager._block)