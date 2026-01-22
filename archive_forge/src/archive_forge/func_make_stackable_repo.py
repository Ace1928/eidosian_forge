import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_stackable_repo(self, relpath='trunk'):
    if isinstance(self.repository_format, remote.RemoteRepositoryFormat):
        repo = self.make_repository(relpath, format='1.9')
        dir = controldir.ControlDir.open(self.get_url(relpath))
        repo = dir.open_repository()
    else:
        repo = self.make_repository(relpath)
    if not repo._format.supports_external_lookups:
        raise tests.TestNotApplicable('format not stackable')
    repo.controldir._format.set_branch_format(bzrbranch.BzrBranchFormat7())
    return repo