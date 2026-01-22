from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_upgrade_repo_with_branches(self):
    control = self.make_repo_with_branches()
    tried, worked, issues = upgrade.smart_upgrade([control], format=self.to_format)
    self.assertLength(3, tried)
    self.assertEqual(tried[0], control)
    self.assertLength(3, worked)
    self.assertEqual(worked[0], control)
    self.assertLength(0, issues)
    self.assertPathExists('repo/backup.bzr.~1~')
    self.assertPathExists('repo/branch1/backup.bzr.~1~')
    self.assertPathExists('repo/branch2/backup.bzr.~1~')
    self.assertEqual(control.open_repository()._format, self.to_format._repository_format)
    b1 = branch.Branch.open('repo/branch1')
    self.assertEqual(b1._format, self.to_format._branch_format)