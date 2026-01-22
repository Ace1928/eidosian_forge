import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
def clone_git_repo(self, from_url, to_url, revision_id=None):
    oldrepos = self.open_git_repo(from_url)
    dir = ControlDir.create(to_url)
    newrepos = dir.create_repository()
    oldrepos.copy_content_into(newrepos, revision_id=revision_id)
    return newrepos