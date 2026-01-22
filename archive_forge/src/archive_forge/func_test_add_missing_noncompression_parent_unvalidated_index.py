import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_add_missing_noncompression_parent_unvalidated_index(self):
    unvalidated = self.make_g_index_missing_parent()
    combined = _mod_index.CombinedGraphIndex([unvalidated])
    index = groupcompress._GCGraphIndex(combined, is_locked=lambda: True, parents=True, track_external_parent_refs=True)
    index.scan_unvalidated_index(unvalidated)
    self.assertEqual(frozenset([(b'missing-parent',)]), index.get_missing_parents())