import contextlib
import errno
import os
import resource
import sys
from breezy import osutils, tests
from breezy.tests import features, script
def test_make_file_big(self):
    self.knownFailure('commit keeps entire files in memory')
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/testfile'])
    tree.add('testfile')
    tree.commit('add small file')
    self.writeBigFile(os.path.join(tree.basedir, 'testfile'))
    tree.commit('small files get big')
    self.knownFailure('commit keeps entire files in memory')