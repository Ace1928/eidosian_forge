from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit.cloud import test_zone
def test_delete_recordset(self):
    fake_zone = test_zone.ZoneTestWrapper(self, zone)
    fake_rs = RecordsetTestWrapper(self, recordset)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['name']]), status_code=404), dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones'], qs_elements=['name={name}'.format(name=fake_zone['name'])]), json={'zones': [fake_zone.get_get_response_json()]}), dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id'], 'recordsets', fake_rs['name']]), status_code=404), dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id'], 'recordsets'], qs_elements=['name={name}'.format(name=fake_rs['name'])]), json={'recordsets': [fake_rs.get_get_response_json()]}), dict(method='DELETE', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', zone['id'], 'recordsets', fake_rs['id']]), status_code=202)])
    self.assertTrue(self.cloud.delete_recordset(fake_zone['name'], fake_rs['name']))
    self.assert_calls()