import os
from .... import tests
from ... import upload
from .. import cmds
def test_disable_auto_push(self):
    self.assertPathDoesNotExist('target')
    self.build_tree(['b'])
    self.wt.add(['b'])
    self.wt.commit('two')
    self.wt.branch.get_config_stack().set('upload_auto', False)
    self.build_tree(['c'])
    self.wt.add(['c'])
    self.wt.commit('three')
    self.assertPathDoesNotExist(os.path.join('target', 'c'))