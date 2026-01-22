from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_get_security_groups(self):
    sot = server.Server(**EXAMPLE)
    response = mock.Mock()
    sgs = [{'description': 'default', 'id': 1, 'name': 'default', 'rules': [{'direction': 'egress', 'ethertype': 'IPv6', 'id': '3c0e45ff-adaf-4124-b083-bf390e5482ff', 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_ip_prefix': None, 'security_group_id': '1', 'project_id': 'e4f50856753b4dc6afee5fa6b9b6c550', 'revision_number': 1, 'tags': ['tag1,tag2'], 'tenant_id': 'e4f50856753b4dc6afee5fa6b9b6c550', 'created_at': '2018-03-19T19:16:56Z', 'updated_at': '2018-03-19T19:16:56Z', 'description': ''}], 'tenant_id': 'e4f50856753b4dc6afee5fa6b9b6c550'}]
    response.status_code = 200
    response.json.return_value = {'security_groups': sgs}
    self.sess.get.return_value = response
    sot.fetch_security_groups(self.sess)
    url = 'servers/IDENTIFIER/os-security-groups'
    self.sess.get.assert_called_with(url)
    self.assertEqual(sot.security_groups, sgs)