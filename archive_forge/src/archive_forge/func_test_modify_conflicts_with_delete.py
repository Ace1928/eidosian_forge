import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_modify_conflicts_with_delete(self):
    builder = self.make_branch_builder('test')
    builder.start_series()
    base_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\nd\ne\n'))])
    other_id = builder.build_snapshot([base_id], [('modify', ('foo', b'a\nc\nd\ne\n'))])
    this_id = builder.build_snapshot([base_id], [('modify', ('foo', b'a\nb2\nc\nd\nX\ne\n'))])
    builder.finish_series()
    branch = builder.get_branch()
    this_tree = branch.controldir.create_workingtree()
    this_tree.lock_write()
    self.addCleanup(this_tree.unlock)
    other_tree = this_tree.controldir.sprout('other', other_id).open_workingtree()
    self.do_merge(this_tree, other_tree)
    if self.merge_type is _mod_merge.LCAMerger:
        self.expectFailure("lca merge doesn't track deleted lines", self.assertFileEqual, 'a\n<<<<<<< TREE\nb2\n=======\n>>>>>>> MERGE-SOURCE\nc\nd\nX\ne\n', 'test/foo')
    else:
        self.assertFileEqual(b'a\n<<<<<<< TREE\nb2\n=======\n>>>>>>> MERGE-SOURCE\nc\nd\nX\ne\n', 'test/foo')