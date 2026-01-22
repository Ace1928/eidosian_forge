import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def example_branch2(self):
    tree = super().example_branch2()
    os.mkdir('checkouts')
    tree = tree.branch.create_checkout('checkouts/branch1')
    os.chdir('checkouts')
    return tree