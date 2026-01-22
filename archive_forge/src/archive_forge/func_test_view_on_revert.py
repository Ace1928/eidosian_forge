from breezy import osutils, tests
def test_view_on_revert(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('revert')
    err_lines = err.splitlines()
    self.assertEqual('Ignoring files outside view. View is a, b', err_lines[0])
    self.assertEqual('-   a', err_lines[1])
    self.assertEqual('-   b', err_lines[2])
    self.assertEqual('', out)