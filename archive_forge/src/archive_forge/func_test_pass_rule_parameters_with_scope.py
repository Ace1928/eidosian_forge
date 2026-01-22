import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_pass_rule_parameters_with_scope(self):
    self.create_config_file('policy.yaml', self.SAMPLE_POLICY_SCOPED)
    self.create_config_file('access.json', jsonutils.dumps(token_fixture.SYSTEM_SCOPED_TOKEN_FIXTURE))
    policy_file = self.get_config_file_fullname('policy.yaml')
    access_file = self.get_config_file_fullname('access.json')
    apply_rule = None
    is_admin = False
    stdout = self._capture_stdout()
    access_data = copy.deepcopy(token_fixture.SYSTEM_SCOPED_TOKEN_FIXTURE['token'])
    access_data['roles'] = [role['name'] for role in access_data['roles']]
    access_data['user_id'] = access_data['user']['id']
    access_data['is_admin'] = is_admin
    shell.tool(policy_file, access_file, apply_rule, is_admin)
    expected = 'passed: sampleservice:sample_rule\npassed: sampleservice:scoped_rule\n'
    self.assertEqual(expected, stdout.getvalue())