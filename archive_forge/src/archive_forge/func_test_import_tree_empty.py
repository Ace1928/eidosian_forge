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
def test_import_tree_empty(self):
    tree = Tree()
    ret, child_modes = import_git_tree(self._texts, self._mapping, b'bla', b'bla', (None, tree.id), None, None, b'somerevid', [], {tree.id: tree}.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
    self.assertEqual(child_modes, {})
    self.assertEqual({(b'git:bla', b'somerevid')}, self._texts.keys())
    self.assertEqual(1, len(ret))
    self.assertEqual(None, ret[0][0])
    self.assertEqual('bla', ret[0][1])
    ie = ret[0][3]
    self.assertEqual('directory', ie.kind)
    self.assertEqual(False, ie.executable)
    self.assertEqual({}, ie.children)
    self.assertEqual(b'somerevid', ie.revision)
    self.assertEqual(None, ie.text_sha1)