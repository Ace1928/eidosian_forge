import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_re_add(self):
    """Test than when a file has 'unintentionally' changed case, we can't
        add a new entry using the new case."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['MixedCase'])
    run_script(self, '\n            $ brz add MixedCase\n            adding MixedCase\n            ')
    osutils.rename('MixedCase', 'mixedcase')
    run_script(self, '\n            $ brz add mixedcase\n            ')