import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_merge_revs_log_single_merge_revision(self):
    wt = self._prepare_tree_with_merges()
    revspec = revisionspec.RevisionSpec.from_string('1.1.1')
    rev = revspec.in_history(wt.branch)
    self.assertFormatterResult(b'      1.1.1 Joe Foo\t2005-11-22\n            rev-merged\n\n', wt.branch, log.ShortLogFormatter, formatter_kwargs=dict(levels=0), show_log_kwargs=dict(start_revision=rev, end_revision=rev))