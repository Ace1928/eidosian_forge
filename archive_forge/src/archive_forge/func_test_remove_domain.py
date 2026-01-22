import mistralclient.auth.keystone
from mistralclient.tests.unit.v2 import base
def test_remove_domain(self):
    params = {'param1': 'p', 'target_param2': 'p2', 'user_domain_param3': 'p3', 'target_project_domain_param4': 'p4'}
    dedomained = self.keystone._remove_domain(params)
    self.assertIn('param1', dedomained)
    self.assertIn('target_param2', dedomained)
    self.assertNotIn('user_domain_param3', dedomained)
    self.assertNotIn('target_project_domain_param4', dedomained)