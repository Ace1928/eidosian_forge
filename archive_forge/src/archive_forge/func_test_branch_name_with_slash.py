from breezy import branchbuilder, errors, tests, urlutils
from breezy.branch import Branch
from breezy.controldir import BranchReferenceLoop
from breezy.tests import per_controldir
from breezy.tests.features import UnicodeFilenameFeature
def test_branch_name_with_slash(self):
    repo = self.make_repository('branch-1')
    try:
        target_branch = self.create_branch(repo.controldir, name='foo/bar')
    except errors.InvalidBranchName:
        raise tests.TestNotApplicable('format does not support branches with / in their name')
    self.assertIn('foo/bar', list(repo.controldir.get_branches()))
    self.assertEqual(target_branch.base, repo.controldir.open_branch(name='foo/bar').base)