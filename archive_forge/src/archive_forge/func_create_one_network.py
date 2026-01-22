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
def create_one_network(attrs=None):
    """Create a fake network.

    :param Dictionary attrs:
        A dictionary with all attributes
    :return:
        An Network object, with id, name, etc.
    """
    attrs = attrs or {}
    network_attrs = {'created_at': '2021-11-29T10:10:23.000000', 'id': 'network-id-' + uuid.uuid4().hex, 'name': 'network-name-' + uuid.uuid4().hex, 'status': 'ACTIVE', 'description': 'network-description-' + uuid.uuid4().hex, 'dns_domain': 'example.org.', 'mtu': '1350', 'project_id': 'project-id-' + uuid.uuid4().hex, 'admin_state_up': True, 'shared': False, 'subnets': ['a', 'b'], 'segments': 'network-segment-' + uuid.uuid4().hex, 'provider:network_type': 'vlan', 'provider:physical_network': 'physnet1', 'provider:segmentation_id': '400', 'router:external': True, 'availability_zones': [], 'availability_zone_hints': [], 'is_default': False, 'is_vlan_transparent': True, 'port_security_enabled': True, 'qos_policy_id': 'qos-policy-id-' + uuid.uuid4().hex, 'ipv4_address_scope': 'ipv4' + uuid.uuid4().hex, 'ipv6_address_scope': 'ipv6' + uuid.uuid4().hex, 'tags': [], 'location': 'MUNCHMUNCHMUNCH', 'updated_at': '2021-11-29T10:10:25.000000'}
    network_attrs.update(attrs)
    network = _network.Network(**network_attrs)
    return network