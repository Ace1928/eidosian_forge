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
def test_tag_annotated(self):
    reva = self.simple_commit_a()
    o = Tag()
    o.name = b'foo'
    o.tagger = b'Jelmer <foo@example.com>'
    o.message = b'add tag'
    o.object = (Commit, reva)
    o.tag_timezone = 0
    o.tag_time = 42
    r = GitRepo('.')
    self.addCleanup(r.close)
    r.object_store.add_object(o)
    r[b'refs/tags/foo'] = o.id
    thebranch = Branch.open('.')
    self.assertEqual({'foo': default_mapping.revision_id_foreign_to_bzr(reva)}, thebranch.tags.get_tag_dict())