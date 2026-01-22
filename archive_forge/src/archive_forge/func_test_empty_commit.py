import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_empty_commit(self):
    wt = self.make_branch_and_tree('.')
    wt.commit('empty commit')
    lf = LogCatcher()
    log.show_log(wt.branch, lf, verbose=True)
    revs = lf.revisions
    self.assertEqual(1, len(revs))
    self.assertEqual('1', revs[0].revno)
    self.assertEqual('empty commit', revs[0].rev.message)
    self.checkDelta(revs[0].delta)