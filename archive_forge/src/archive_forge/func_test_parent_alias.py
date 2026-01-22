from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_parent_alias(self):
    t = self.make_branch_and_tree('base')
    t.branch.get_config_stack().set('test', 'base')
    clone = t.branch.controldir.sprout('clone').open_branch()
    clone.get_config_stack().set('test', 'clone')
    out, err = self.run_bzr(['config', '-d', ':parent', 'test'], working_dir='clone')
    self.assertEqual('base\n', out)