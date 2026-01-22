import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
def test_bzr_branch_bound_to_git(self):
    path, (gitsha1, gitsha2) = self.make_tworev_branch()
    wt = Branch.open(path).create_checkout('co')
    self.build_tree_contents([('co/foobar', b'blah')])
    self.assertRaises(errors.NoRoundtrippingSupport, wt.commit, 'commit from bound branch.')
    revid = wt.commit('commit from bound branch.', lossy=True)
    self.assertEqual(revid, wt.branch.last_revision())
    self.assertEqual(revid, wt.branch.get_master_branch().last_revision())