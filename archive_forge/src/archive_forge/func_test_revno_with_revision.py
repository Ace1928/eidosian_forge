import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_revno_with_revision(self):
    wt = self.make_branch_and_tree('.')
    revid1 = wt.commit('rev1')
    revid2 = wt.commit('rev2')
    out, err = self.run_bzr('revno -r-2 .')
    self.assertEqual('1\n', out)
    out, err = self.run_bzr('revno -rrevid:%s .' % revid1.decode('utf-8'))
    self.assertEqual('1\n', out)