from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_register_cli_options(self):
    f = self._make_fixture()
    opt1 = cfg.StrOpt('first_test_opt', default='initial_value_1')
    opt2 = cfg.StrOpt('second_test_opt', default='initial_value_2')
    f.register_cli_opts([opt1, opt2])
    self.assertEqual(f.conf.get('first_test_opt'), opt1.default)
    self.assertEqual(f.conf.get('second_test_opt'), opt2.default)