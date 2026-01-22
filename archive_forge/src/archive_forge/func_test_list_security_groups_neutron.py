import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_security_groups_neutron(self):
    project_id = 42
    self.cloud.secgroup_source = 'neutron'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups'], qs_elements=['project_id=%s' % project_id]), json={'security_groups': [neutron_grp_dict]})])
    self.cloud.list_security_groups(filters={'project_id': project_id})
    self.assert_calls()