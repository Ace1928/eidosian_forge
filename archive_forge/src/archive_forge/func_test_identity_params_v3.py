from openstack import exceptions
from openstack.tests.unit import base
def test_identity_params_v3(self):
    project_data = self._get_project_data(v3=True)
    self.register_uris([dict(method='GET', uri='https://identity.example.com/v3/projects', json=dict(projects=[project_data.json_response['project']]))])
    ret = self.cloud._get_identity_params(domain_id='5678', project=project_data.project_name)
    self.assertIn('default_project_id', ret)
    self.assertEqual(ret['default_project_id'], project_data.project_id)
    self.assertIn('domain_id', ret)
    self.assertEqual(ret['domain_id'], '5678')
    self.assert_calls()