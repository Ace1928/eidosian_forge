import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def prepare_tree(self):
    tree = self.make_branch_and_tree('')
    self.build_tree(['/hello.txt', '/goodbye.txt'])
    tree.add('hello.txt')
    tree.commit(message='message1', committer='committer1', authors=['author1'])
    tree.add('goodbye.txt')
    tree.commit(message='message2', committer='committer2', authors=['author2'])