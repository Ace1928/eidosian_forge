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
def clone_git_branch(self, from_url, to_url):
    from_dir = ControlDir.open(from_url)
    to_dir = from_dir.sprout(to_url)
    return to_dir.open_branch()