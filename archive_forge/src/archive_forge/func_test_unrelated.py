import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_unrelated(self):
    os.chdir('..')
    os.mkdir('unrelated')
    os.chdir('unrelated')
    self.run_bzr('init')
    self.make_file('hello', 'hi world')
    self.run_bzr('add')
    self.run_bzr('commit -m x')
    self.run_bzr_error(['brz: ERROR: Branches have no common ancestor, and no merge base.*'], ['rebase', '../main'])