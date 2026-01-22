import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_as_revision_id_uncommitted(self):
    spec = RevisionSpec.from_string('annotate:annotate-tree/file1:3')
    e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
    self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/file1:3\\' does not exist in branch: .*\nLine 3 has not been committed.")