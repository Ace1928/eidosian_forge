import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_revno_tree(self):
    wt = self.make_branch_and_tree('branch')
    checkout = wt.branch.create_checkout('checkout', lightweight=True)
    self.build_tree(['branch/file'])
    wt.add(['file'])
    wt.commit('mkfile')
    out, err = self.run_bzr('revno checkout')
    self.assertEqual('', err)
    self.assertEqual('1\n', out)
    out, err = self.run_bzr('revno --tree checkout')
    self.assertEqual('', err)
    self.assertEqual('0\n', out)