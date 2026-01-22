import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_unshared_repository_with_colocated_branches(self):
    format = controldir.format_registry.make_controldir('development-colo')
    transport = self.get_transport()
    repo = self.make_repository('repo', shared=False, format=format)
    repo.set_make_working_trees(True)
    repo.controldir.create_branch(name='foo')
    out, err = self.run_bzr('info repo')
    self.assertEqualDiff('Unshared repository with trees and colocated branches (format: development-colo)\nLocation:\n  repository: repo\n', out)
    self.assertEqual('', err)