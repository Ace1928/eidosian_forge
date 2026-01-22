from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_branch_to_lightweight_checkout_failure(self):
    parent, child, reconfiguration = self.prepare_branch_to_lightweight_checkout()
    old_Repository_fetch = vf_repository.VersionedFileRepository.fetch
    vf_repository.VersionedFileRepository.fetch = None
    try:
        self.assertRaises(TypeError, reconfiguration.apply)
    finally:
        vf_repository.VersionedFileRepository.fetch = old_Repository_fetch
    child = _mod_branch.Branch.open('child')
    self.assertContainsRe(child.base, 'child/$')