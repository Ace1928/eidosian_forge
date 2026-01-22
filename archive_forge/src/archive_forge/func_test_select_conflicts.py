import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def test_select_conflicts(self):
    tree = self.make_branch_and_tree('.')
    clist = bzr_conflicts.ConflictList

    def check_select(not_selected, selected, paths, **kwargs):
        self.assertEqual((not_selected, selected), tree_conflicts.select_conflicts(tree, paths, **kwargs))
    foo = bzr_conflicts.ContentsConflict('foo')
    bar = bzr_conflicts.ContentsConflict('bar')
    tree_conflicts = clist([foo, bar])
    check_select(clist([bar]), clist([foo]), ['foo'])
    check_select(clist(), tree_conflicts, [''], ignore_misses=True, recurse=True)
    foobaz = bzr_conflicts.ContentsConflict('foo/baz')
    tree_conflicts = clist([foobaz, bar])
    check_select(clist([bar]), clist([foobaz]), ['foo'], ignore_misses=True, recurse=True)
    qux = bzr_conflicts.PathConflict('qux', 'foo/baz')
    tree_conflicts = clist([qux])
    check_select(clist(), tree_conflicts, ['foo'], ignore_misses=True, recurse=True)
    check_select(tree_conflicts, clist(), ['foo'], ignore_misses=True)