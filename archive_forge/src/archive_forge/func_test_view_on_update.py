from breezy import osutils, tests
def test_view_on_update(self):
    tree_1, tree_2 = self.make_abc_tree_and_clone_with_ab_view()
    self.run_bzr('bind ../tree_1', working_dir='tree_2')
    out, err = self.run_bzr('update', working_dir='tree_2')
    self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\n" % osutils.pathjoin(self.test_dir, 'tree_1'), err)
    self.assertEqual('', out)