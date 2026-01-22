import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_store_and_restore_uncommitted(self):
    checkout = self.prepare()
    self.run_bzr(['switch', '--store', '-d', 'checkout', 'new'])
    self.build_tree(['checkout/b'])
    checkout.add('b')
    self.assertPathDoesNotExist('checkout/a')
    self.assertPathExists('checkout/b')
    self.run_bzr(['switch', '--store', '-d', 'checkout', 'orig'])
    self.assertPathExists('checkout/a')
    self.assertPathDoesNotExist('checkout/b')