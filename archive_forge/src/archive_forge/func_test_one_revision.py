import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_one_revision(self):
    tree = self.setup_ab_tree()
    lf = LogCatcher()
    rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
    log.show_log(tree.branch, lf, verbose=True, start_revision=rev, end_revision=rev)
    self.assertEqual(1, len(lf.revisions))
    self.assertEqual(None, lf.revisions[0].revno)
    self.assertEqual(b'3a', lf.revisions[0].rev.revision_id)