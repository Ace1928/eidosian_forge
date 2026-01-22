import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_ci(self):
    wt = self._make_mixed_case_tree()
    self.run_bzr('add')
    got = self.run_bzr('ci -m message camelcaseparent LOWERCASEPARENT')[1]
    for expected in ['CamelCaseParent', 'lowercaseparent', 'CamelCaseParent/CamelCase', 'lowercaseparent/lowercase']:
        self.assertContainsRe(got, 'added ' + expected + '\n')