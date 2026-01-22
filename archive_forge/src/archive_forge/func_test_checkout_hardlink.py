import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_hardlink(self):
    self.requireFeature(HardlinkFeature(self.test_dir))
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/file1'])
    source.add('file1')
    source.commit('added file')
    out, err = self.run_bzr('checkout source target --hardlink')
    source_stat = os.stat('source/file1')
    target_stat = os.stat('target/file1')
    self.assertEqual(source_stat, target_stat)