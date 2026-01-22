import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
@mock.patch.object(policy, 'LOG')
def test_warning_on_redundant_file_rules(self, mock_log):
    rules = yaml.dump({'admin': 'is_admin:True'})
    self.create_config_file('policy.yaml', rules)
    path = self.get_config_file_fullname('policy.yaml')
    enforcer = policy.Enforcer(self.conf, policy_file=path)
    enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:True'))
    enforcer.load_rules(True)
    warn_msg = 'Policy Rules %(names)s specified in policy files are the same as the defaults provided by the service. You can remove these rules from policy files which will make maintenance easier. You can detect these redundant rules by ``oslopolicy-list-redundant`` tool also.'
    mock_log.warning.assert_any_call(warn_msg, {'names': ['admin']})