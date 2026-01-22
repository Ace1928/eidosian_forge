from io import StringIO
from .. import config
from .. import status as _mod_status
from ..revisionspec import RevisionSpec
from ..status import show_pending_merges, show_tree_status
from . import TestCaseWithTransport
def make_multiple_pending_tree(self):
    config.GlobalStack().set('email', 'Joe Foo <joe@foo.com>')
    tree = self.make_branch_and_tree('a')
    tree.commit('commit 1', timestamp=1196796819, timezone=0)
    tree2 = tree.controldir.clone('b').open_workingtree()
    tree.commit('commit 2', timestamp=1196796819, timezone=0)
    tree2.commit('commit 2b', timestamp=1196796819, timezone=0)
    tree3 = tree2.controldir.clone('c').open_workingtree()
    tree2.commit('commit 3b', timestamp=1196796819, timezone=0)
    tree3.commit('commit 3c', timestamp=1196796819, timezone=0)
    tree.merge_from_branch(tree2.branch)
    tree.merge_from_branch(tree3.branch, force=True)
    return tree