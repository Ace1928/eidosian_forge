from breezy import osutils, tests
def make_abc_tree_and_clone_with_ab_view(self):
    wt1 = self.make_branch_and_tree('tree_1')
    self.build_tree(['tree_1/a', 'tree_1/b', 'tree_1/c'])
    wt1.add(['a', 'b', 'c'])
    wt1.commit('adding a b c')
    wt2 = wt1.controldir.sprout('tree_2').open_workingtree()
    wt2.views.set_view('my', ['a', 'b'])
    self.build_tree_contents([('tree_1/a', b'changed a\n'), ('tree_1/c', b'changed c\n')])
    wt1.commit('changing a c')
    return (wt1, wt2)