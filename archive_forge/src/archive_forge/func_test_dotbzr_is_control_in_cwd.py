from breezy.osutils import basename
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_dotbzr_is_control_in_cwd(self):
    tree = self.make_branch_and_tree('.')
    self.validate_tree_is_controlfilename(tree)