import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def get_text_by_path(tree, path):
    return tree.get_file_text(path)