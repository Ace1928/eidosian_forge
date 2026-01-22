from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_set_default_group(self):
    f = self._make_fixture()
    opt = cfg.StrOpt('new_test_opt', default='initial_value')
    f.conf.register_opt(opt, group='foo')
    f.set_default(name='new_test_opt', default='alternate_value', group='foo')
    self.assertEqual('alternate_value', f.conf.foo.new_test_opt)
    f.cleanUp()
    self.assertEqual('initial_value', f.conf.foo.new_test_opt)