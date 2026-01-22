import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_mv_newname_exists_after(self):
    wt = self._make_mixed_case_tree()
    self.run_bzr('add')
    self.run_bzr('ci -m message')
    os.unlink('CamelCaseParent/CamelCase')
    osutils.rename('lowercaseparent/lowercase', 'lowercaseparent/LOWERCASE')
    run_script(self, '\n            $ brz mv --after camelcaseparent/camelcase LOWERCASEPARENT/LOWERCASE\n            2>brz: ERROR: Could not move CamelCase => lowercase: lowercaseparent/lowercase is already versioned.\n            ')