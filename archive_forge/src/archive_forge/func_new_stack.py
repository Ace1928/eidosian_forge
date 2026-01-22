import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def new_stack(tree, path=None, file_id=None):
    if path.endswith('.txt'):
        return [ContentFilter(_swapcase, _swapcase)]
    else:
        return []