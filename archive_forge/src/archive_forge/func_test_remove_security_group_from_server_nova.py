import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_remove_security_group_from_server_nova(self):
    self.has_neutron = False
    self.cloud.secgroup_source = 'nova'
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': [nova_grp_dict]}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri='%s/servers/%s/action' % (fakes.COMPUTE_ENDPOINT, '1234'), validate=dict(json={'removeSecurityGroup': {'name': 'nova-sec-group'}}))])
    ret = self.cloud.remove_server_security_groups(dict(id='1234'), 'nova-sec-group')
    self.assertTrue(ret)
    self.assert_calls()