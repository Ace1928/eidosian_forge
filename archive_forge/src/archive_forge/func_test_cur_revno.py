import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_cur_revno(self):
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    lf = LogCatcher()
    wt.commit('empty commit')
    log.show_log(b, lf, verbose=True, start_revision=1, end_revision=1)
    self.assertInvalidRevisonNumber(b, 2, 1)
    self.assertInvalidRevisonNumber(b, 1, 2)
    self.assertInvalidRevisonNumber(b, 0, 2)
    self.assertInvalidRevisonNumber(b, -1, 1)
    self.assertInvalidRevisonNumber(b, 1, -1)
    self.assertInvalidRevisonNumber(b, 1, 0)