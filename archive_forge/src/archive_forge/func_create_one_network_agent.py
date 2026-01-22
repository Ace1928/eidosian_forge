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
def create_one_network_agent(attrs=None):
    """Create a fake network agent

    :param Dictionary attrs:
        A dictionary with all attributes
    :return:
        An Agent object, with id, agent_type, and so on.
    """
    attrs = attrs or {}
    agent_attrs = {'id': 'agent-id-' + uuid.uuid4().hex, 'agent_type': 'agent-type-' + uuid.uuid4().hex, 'host': 'host-' + uuid.uuid4().hex, 'availability_zone': 'zone-' + uuid.uuid4().hex, 'alive': True, 'admin_state_up': True, 'binary': 'binary-' + uuid.uuid4().hex, 'configurations': {'subnet': 2, 'networks': 1}, 'location': 'MUNCHMUNCHMUNCH'}
    agent_attrs.update(attrs)
    agent = network_agent.Agent(**agent_attrs)
    return agent