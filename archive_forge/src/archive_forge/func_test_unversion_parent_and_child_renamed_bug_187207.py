from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_parent_and_child_renamed_bug_187207(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['del/', 'del/sub/', 'del/sub/b'])
    tree.add(['del', 'del/sub', 'del/sub/b'])
    revid = tree.commit('setup')
    revtree = tree.branch.repository.revision_tree(revid)
    tree.rename_one('del/sub', 'sub')
    self.assertThat(tree, HasPathRelations(revtree, [('', ''), ('del/', 'del/'), ('sub/', 'del/sub/'), ('sub/b', 'del/sub/b')]))
    if tree.has_versioned_directories():
        tree.unversion(['del', 'sub/b'])
    else:
        tree.unversion(['sub/b'])
    self.assertThat(tree, HasPathRelations(revtree, [('', ''), ('sub/', 'del/sub/')]))