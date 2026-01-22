import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_nova_egress_security_group_rule(self):
    self.has_neutron = False
    self.cloud.secgroup_source = 'nova'
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': [nova_grp_dict]})])
    self.assertRaises(exceptions.SDKException, self.cloud.create_security_group_rule, secgroup_name_or_id='nova-sec-group', direction='egress')
    self.assert_calls()