from breezy import errors, tests, workingtree
def test_subsume_failure(self):
    base_tree, sub_tree = self.make_trees()
    if base_tree.path2id('') == sub_tree.path2id(''):
        raise tests.TestSkipped('This test requires unique roots')
    self.assertRaises(errors.BadSubsumeSource, base_tree.subsume, base_tree)
    self.assertRaises(errors.BadSubsumeSource, sub_tree.subsume, base_tree)
    self.build_tree(['subtree2/'])
    sub_tree2 = self.make_branch_and_tree('subtree2')
    self.assertRaises(errors.BadSubsumeSource, sub_tree.subsume, sub_tree2)
    self.build_tree(['tree/subtree/subtree3/'])
    sub_tree3 = self.make_branch_and_tree('tree/subtree/subtree3')
    self.assertRaises(errors.BadSubsumeSource, base_tree.subsume, sub_tree3)