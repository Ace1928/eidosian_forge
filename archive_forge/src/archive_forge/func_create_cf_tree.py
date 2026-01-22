import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def create_cf_tree(self, txt_reader, txt_writer, dir='.'):
    tree = self.make_branch_and_tree(dir)

    def _content_filter_stack(path=None, file_id=None):
        if path.endswith('.txt'):
            return [ContentFilter(txt_reader, txt_writer)]
        else:
            return []
    tree._content_filter_stack = _content_filter_stack
    self.build_tree_contents([(dir + '/file1.txt', b'Foo Txt'), (dir + '/file2.bin', b'Foo Bin')])
    tree.add(['file1.txt', 'file2.bin'])
    tree.commit('commit raw content')
    return (tree, 'file1.txt', 'file2.bin')