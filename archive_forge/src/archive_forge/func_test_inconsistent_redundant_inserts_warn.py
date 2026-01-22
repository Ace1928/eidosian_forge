import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_inconsistent_redundant_inserts_warn(self):
    """Should not insert a record that is already present."""
    warnings = []

    def warning(template, args):
        warnings.append(template % args)
    _trace_warning = trace.warning
    trace.warning = warning
    try:
        self.do_inconsistent_inserts(inconsistency_fatal=False)
    finally:
        trace.warning = _trace_warning
    self.assertContainsRe('\n'.join(warnings), "^inconsistent details in skipped record: \\(b?'b',\\) \\(b?'42 32 0 8', \\(\\(\\),\\)\\) \\(b?'74 32 0 8', \\(\\(\\(b?'a',\\),\\),\\)\\)$")