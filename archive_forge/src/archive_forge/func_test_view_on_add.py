from breezy import osutils, tests
def test_view_on_add(self):
    wt = self.make_abc_tree_with_ab_view()
    out, err = self.run_bzr('add')
    self.assertEqual('Ignoring files outside view. View is a, b\n', err)
    self.assertEqual('adding a\nadding b\n', out)