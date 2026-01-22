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
def test_last_revision_info(self):
    reva = self.simple_commit_a()
    self.build_tree(['b'])
    r = GitRepo('.')
    self.addCleanup(r.close)
    r.stage('b')
    revb = r.do_commit(b'b', committer=b'Somebody <foo@example.com>')
    thebranch = Branch.open('.')
    self.assertEqual((2, default_mapping.revision_id_foreign_to_bzr(revb)), thebranch.last_revision_info())