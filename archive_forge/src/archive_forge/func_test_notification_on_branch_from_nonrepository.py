import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_notification_on_branch_from_nonrepository(self):
    fmt = BzrDirMetaFormat1()
    t = self.get_transport()
    t.mkdir('a')
    dir = fmt.initialize_on_transport(t.clone('a'))
    self.assertRaises(errors.NoRepositoryPresent, dir.open_repository)
    e = self.assertRaises(errors.NotBranchError, dir.open_branch)
    self.assertNotContainsRe(str(e), 'location is a repository')