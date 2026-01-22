import os
import signal
import sys
import time
from breezy import debug, tests
def test_bytes_reports_activity(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/one'])
    tree.add('one')
    rev_id = tree.commit('first')
    remote_trans = self.make_smart_server('.')
    out, err = self.run_brz_subprocess('branch -Dbytes -Oprogress_bar=text %s/tree target' % (remote_trans.base,))
    self.assertContainsRe(err, b'Branched 1 revision')
    self.assertContainsRe(err, b'Transferred:.*kB')