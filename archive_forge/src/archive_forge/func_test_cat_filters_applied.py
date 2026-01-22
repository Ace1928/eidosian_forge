from ... import tests
from ...transport import memory
def test_cat_filters_applied(self):
    from ...tree import Tree
    from ..test_filters import _stack_2
    wt = self.make_branch_and_tree('.')
    self.build_tree_contents([('README', b'junk\nline 1 of README\nline 2 of README\n')])
    wt.add('README')
    wt.commit('Making sure there is a basis_tree available')
    url = self.get_readonly_url() + '/README'
    real_content_filter_stack = Tree._content_filter_stack

    def _custom_content_filter_stack(tree, path=None, file_id=None):
        return _stack_2
    Tree._content_filter_stack = _custom_content_filter_stack
    try:
        out, err = self.run_bzr(['cat', url, '--filters'])
        self.assertEqual('LINE 1 OF readme\nLINE 2 OF readme\n', out)
        self.assertEqual('', err)
    finally:
        Tree._content_filter_stack = real_content_filter_stack