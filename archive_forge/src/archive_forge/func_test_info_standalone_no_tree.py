import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_standalone_no_tree(self):
    format = controldir.format_registry.make_controldir('default')
    branch = self.make_branch('branch')
    repo = branch.repository
    out, err = self.run_bzr('info branch -v')
    self.assertEqualDiff('Standalone branch (format: {})\nLocation:\n  branch root: branch\n\nFormat:\n       control: Meta directory format 1\n        branch: {}\n    repository: {}\n\nControl directory:\n         1 branches\n\nBranch history:\n         0 revisions\n\nRepository:\n         0 revisions\n'.format(info.describe_format(repo.controldir, repo, branch, None), format.get_branch_format().get_format_description(), format.repository_format.get_format_description()), out)
    self.assertEqual('', err)