import os
from .... import tests
from ... import upload
from .. import cmds
def make_start_branch(self):
    self.wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    self.wt.add(['a'])
    self.wt.commit('one')