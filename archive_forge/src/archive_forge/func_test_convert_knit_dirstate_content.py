from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_convert_knit_dirstate_content(self):
    tree = self.make_branch_and_tree('tree', format='knit')
    self.build_tree(['tree/file'])
    tree.add(['file'])
    target = controldir.format_registry.make_controldir('dirstate')
    converter = tree.controldir._format.get_converter(target)
    converter.convert(tree.controldir, None)
    new_tree = workingtree.WorkingTree.open('tree')
    self.assertIs(new_tree.__class__, workingtree_4.WorkingTree4)
    self.assertEqual(b'null:', new_tree.last_revision())