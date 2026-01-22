import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_mv_newname_root(self):
    wt = self._make_mixed_case_tree()
    self.run_bzr('add')
    self.run_bzr('ci -m message')
    run_script(self, '\n            $ brz mv camelcaseparent NewCamelCaseParent\n            CamelCaseParent => NewCamelCaseParent\n            ')