import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
def test_merge3_will_detect_cherrypick(self):
    this_tree = self.make_branch_and_tree('this')
    self.build_tree_contents([('this/file', b'a\n')])
    this_tree.add('file')
    this_tree.commit('rev1')
    other_tree = this_tree.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/file', b'a\nb\n')])
    other_tree.commit('rev2b', rev_id=b'rev2b')
    self.build_tree_contents([('other/file', b'a\nb\nc\n')])
    other_tree.commit('rev3b', rev_id=b'rev3b')
    this_tree.lock_write()
    self.addCleanup(this_tree.unlock)
    merger = _mod_merge.Merger.from_revision_ids(this_tree, b'rev3b', b'rev2b', other_tree.branch)
    merger.merge_type = _mod_merge.Merge3Merger
    merger.do_merge()
    self.assertFileEqual(b'a\n<<<<<<< TREE\n=======\nc\n>>>>>>> MERGE-SOURCE\n', 'this/file')