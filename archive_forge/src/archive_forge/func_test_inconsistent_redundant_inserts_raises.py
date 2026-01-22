import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_inconsistent_redundant_inserts_raises(self):
    e = self.assertRaises(knit.KnitCorrupt, self.do_inconsistent_inserts, inconsistency_fatal=True)
    self.assertContainsRe(str(e), "Knit.* corrupt: inconsistent details in add_records: \\(b?'b',\\) \\(b?'42 32 0 8', \\(\\(\\),\\)\\) \\(b?'74 32 0 8', \\(\\(\\(b?'a',\\),\\),\\)\\)")