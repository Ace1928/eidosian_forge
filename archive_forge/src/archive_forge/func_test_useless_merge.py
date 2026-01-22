import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_useless_merge(self):
    self.make_file('bar', '42')
    self.run_bzr('add')
    self.run_bzr('commit -m that')
    os.chdir('../feature')
    self.make_file('hello', 'my data')
    self.run_bzr('commit -m this')
    self.run_bzr('merge')
    self.run_bzr('commit -m merge')
    self.run_bzr('rebase')