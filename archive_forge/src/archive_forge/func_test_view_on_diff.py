from breezy import osutils, tests
def test_view_on_diff(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('diff', retcode=1)
    self.assertEqual('*** Ignoring files outside view. View is a, b\n', err)