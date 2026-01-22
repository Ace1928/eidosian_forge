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
def test_import_tree_with_file(self):
    blob = Blob.from_string(b'bar1')
    tree = Tree()
    tree.add(b'foo', stat.S_IFREG | 420, blob.id)
    objects = {blob.id: blob, tree.id: tree}
    ret, child_modes = import_git_tree(self._texts, self._mapping, b'bla', b'bla', (None, tree.id), None, None, b'somerevid', [], objects.__getitem__, (None, stat.S_IFDIR), DummyStoreUpdater(), self._mapping.generate_file_id)
    self.assertEqual(child_modes, {})
    self.assertEqual(2, len(ret))
    self.assertEqual(None, ret[0][0])
    self.assertEqual('bla', ret[0][1])
    self.assertEqual(None, ret[1][0])
    self.assertEqual('bla/foo', ret[1][1])
    ie = ret[0][3]
    self.assertEqual('directory', ie.kind)
    ie = ret[1][3]
    self.assertEqual('file', ie.kind)
    self.assertEqual(b'git:bla/foo', ie.file_id)
    self.assertEqual(b'somerevid', ie.revision)
    self.assertEqual(osutils.sha_strings([b'bar1']), ie.text_sha1)
    self.assertEqual(False, ie.executable)