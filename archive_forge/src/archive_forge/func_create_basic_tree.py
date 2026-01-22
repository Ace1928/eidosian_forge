import os
import sys
import time
from breezy import tests
from breezy.bzr import hashcache
from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def create_basic_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a', 'tree/b/', 'tree/b/c'])
    tree.add(['a', 'b', 'b/c'])
    tree.commit('creating an initial tree.')
    return tree