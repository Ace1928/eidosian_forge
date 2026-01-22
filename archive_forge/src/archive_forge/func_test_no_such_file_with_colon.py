import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_no_such_file_with_colon(self):
    spec = RevisionSpec.from_string('annotate:annotate-tree/fi:le2:1')
    e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
    self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/fi:le2:1\\' does not exist in branch: .*\nFile 'fi:le2' is not versioned")