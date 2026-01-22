import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def reopen_repo(self, repo):
    same_repo = repo.controldir.open_repository()
    same_repo.lock_write()
    self.addCleanup(same_repo.unlock)
    return same_repo