from breezy import osutils, tests
def make_abc_tree_with_ab_view(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'c'])
    wt.views.set_view('my', ['a', 'b'])
    return wt