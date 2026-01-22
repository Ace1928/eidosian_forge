import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_add_volume_type_access(self):
    volume_type = dict(id='voltype01', description='volume type description', name='name', is_public=False)
    project_001 = dict(volume_type_id='voltype01', name='name', project_id='prj01')
    project_002 = dict(volume_type_id='voltype01', name='name', project_id='prj02')
    volume_type_access = [project_001, project_002]
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types']), json={'volume_types': [volume_type]}), dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['types', volume_type['id'], 'action']), json={'addProjectAccess': {'project': project_002['project_id']}}, validate=dict(json={'addProjectAccess': {'project': project_002['project_id']}})), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types']), json={'volume_types': [volume_type]}), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['types', volume_type['id'], 'os-volume-type-access']), json={'volume_type_access': volume_type_access})])
    self.cloud.add_volume_type_access(volume_type['name'], project_002['project_id'])
    self.assertEqual(len(self.cloud.get_volume_type_access(volume_type['name'])), 2)
    self.assert_calls()