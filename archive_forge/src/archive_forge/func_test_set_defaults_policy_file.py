import copy
from oslo_config import cfg
from oslotest import base as test_base
from oslo_policy import opts
def test_set_defaults_policy_file(self):
    opts._register(self.conf)
    self.assertNotEqual('new-value.json', self.conf.oslo_policy.policy_file)
    opts.set_defaults(self.conf, policy_file='new-value.json')
    self.assertEqual('new-value.json', self.conf.oslo_policy.policy_file)