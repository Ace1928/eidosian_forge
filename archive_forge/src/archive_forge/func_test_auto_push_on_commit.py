import os
from .... import tests
from ... import upload
from .. import cmds
def test_auto_push_on_commit(self):
    self.assertPathDoesNotExist('target')
    self.build_tree(['b'])
    self.wt.add(['b'])
    self.wt.commit('two')
    self.assertPathExists('target')
    self.assertPathExists(os.path.join('target', 'a'))
    self.assertPathExists(os.path.join('target', 'b'))