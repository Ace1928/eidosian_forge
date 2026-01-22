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
def test_becomes_executable(self):
    self.make_git_repo('d')
    os.chdir('d')
    bb = GitBranchBuilder()
    bb.set_file('foobar', b'foo\nbar\n', False)
    mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    bb.set_file('foobar', b'foo\nbar\n', True)
    mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    gitsha2 = bb.finish()[mark2]
    os.chdir('..')
    oldrepo = self.open_git_repo('d')
    newrepo = self.clone_git_repo('d', 'f')
    revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
    tree = newrepo.revision_tree(revid)
    self.assertTrue(tree.has_filename('foobar'))
    self.assertEqual(True, tree.is_executable('foobar'))
    self.assertEqual(revid, tree.get_file_revision('foobar'))