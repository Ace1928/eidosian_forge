import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def test_resolve_conflict_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('hello', b'hello world4'), ('hello.THIS', b'hello world2'), ('hello.BASE', b'hello world1')])
    os.mkdir('hello.OTHER')
    tree.add('hello', ids=b'q')
    l = bzr_conflicts.ConflictList([bzr_conflicts.TextConflict('hello')])
    l.remove_files(tree)