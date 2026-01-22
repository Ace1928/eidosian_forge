import argparse
import copy
from random import choice
from random import randint
from unittest import mock
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import address_group as _address_group
from openstack.network.v2 import address_scope as _address_scope
from openstack.network.v2 import agent as network_agent
from openstack.network.v2 import auto_allocated_topology as allocated_topology
from openstack.network.v2 import availability_zone as _availability_zone
from openstack.network.v2 import extension as _extension
from openstack.network.v2 import flavor as _flavor
from openstack.network.v2 import local_ip as _local_ip
from openstack.network.v2 import local_ip_association as _local_ip_association
from openstack.network.v2 import ndp_proxy as _ndp_proxy
from openstack.network.v2 import network as _network
from openstack.network.v2 import network_ip_availability as _ip_availability
from openstack.network.v2 import network_segment_range as _segment_range
from openstack.network.v2 import port as _port
from openstack.network.v2 import rbac_policy as network_rbac
from openstack.network.v2 import segment as _segment
from openstack.network.v2 import service_profile as _flavor_profile
from openstack.network.v2 import trunk as _trunk
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit import utils
@staticmethod
def create_one_subnet(attrs=None):
    """Create a fake subnet.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object faking the subnet
        """
    attrs = attrs or {}
    project_id = 'project-id-' + uuid.uuid4().hex
    subnet_attrs = {'id': 'subnet-id-' + uuid.uuid4().hex, 'name': 'subnet-name-' + uuid.uuid4().hex, 'network_id': 'network-id-' + uuid.uuid4().hex, 'cidr': '10.10.10.0/24', 'project_id': project_id, 'enable_dhcp': True, 'dns_nameservers': [], 'allocation_pools': [], 'host_routes': [], 'ip_version': 4, 'gateway_ip': '10.10.10.1', 'ipv6_address_mode': None, 'ipv6_ra_mode': None, 'segment_id': None, 'service_types': [], 'subnetpool_id': None, 'description': 'subnet-description-' + uuid.uuid4().hex, 'tags': [], 'location': 'MUNCHMUNCHMUNCH'}
    subnet_attrs.update(attrs)
    subnet = fakes.FakeResource(info=copy.deepcopy(subnet_attrs), loaded=True)
    subnet.is_dhcp_enabled = subnet_attrs['enable_dhcp']
    subnet.subnet_pool_id = subnet_attrs['subnetpool_id']
    return subnet