import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def test_info_revinfo_optional(self):
    tree = self.make_branch_and_tree('.')

    def last_revision_info(self):
        raise errors.UnsupportedOperation(last_revision_info, self)
    self.overrideAttr(branch.Branch, 'last_revision_info', last_revision_info)
    out, err = self.run_bzr('info -v .')
    self.assertEqual('Standalone tree (format: 2a)\nLocation:\n  branch root: .\n\nFormat:\n       control: Meta directory format 1\n  working tree: Working tree format 6\n        branch: Branch format 7\n    repository: Repository format 2a - rich roots, group compression and chk inventories\n\nControl directory:\n         1 branches\n\nIn the working tree:\n         0 unchanged\n         0 modified\n         0 added\n         0 removed\n         0 renamed\n         0 copied\n         0 unknown\n         0 ignored\n         0 versioned subdirectories\n', out)
    self.assertEqual('', err)