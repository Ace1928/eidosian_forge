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
def test_sprouted_ghost_tags(self):
    path, gitsha = self.make_onerev_branch()
    r = GitRepo(path)
    self.addCleanup(r.close)
    r.refs[b'refs/tags/lala'] = b'aa' * 20
    oldrepo = Repository.open(path)
    revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
    warnings, newbranch = self.callCatchWarnings(self.clone_git_branch, path, 'f')
    self.assertEqual({}, newbranch.tags.get_tag_dict())
    self.assertIn(('ref refs/tags/lala points at non-present sha ' + 'aa' * 20,), [w.args for w in warnings])