import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_selected_file_merge_commit(self):
    """Ensure the correct error is raised"""
    tree = self.make_branch_and_tree('foo')
    tree.commit('commit 1')
    tree.add_parent_tree_id(b'example')
    self.build_tree(['foo/bar', 'foo/baz'])
    tree.add(['bar', 'baz'])
    err = self.assertRaises(CannotCommitSelectedFileMerge, tree.commit, 'commit 2', specific_files=['bar', 'baz'])
    self.assertEqual(['bar', 'baz'], err.files)
    self.assertEqual('Selected-file commit of merges is not supported yet: files bar, baz', str(err))