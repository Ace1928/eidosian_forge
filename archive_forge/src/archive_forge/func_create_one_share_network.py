import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_share_network(attrs=None, methods=None):
    """Create a fake share network

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_network = {'id': str(uuid.uuid4()), 'project_id': uuid.uuid4().hex, 'created_at': datetime.datetime.now().isoformat(), 'description': 'description-' + uuid.uuid4().hex, 'name': 'name-' + uuid.uuid4().hex, 'status': 'active', 'security_service_update_support': True, 'share_network_subnets': [{'id': str(uuid.uuid4()), 'availability_zone': None, 'created_at': datetime.datetime.now().isoformat(), 'updated_at': datetime.datetime.now().isoformat(), 'segmentation_id': 1010, 'neutron_net_id': str(uuid.uuid4()), 'neutron_subnet_id': str(uuid.uuid4()), 'ip_version': 4, 'cidr': '10.0.0.0/24', 'network_type': 'vlan', 'mtu': '1500', 'gateway': '10.0.0.1'}]}
    share_network.update(attrs)
    share_network = osc_fakes.FakeResource(info=copy.deepcopy(share_network), methods=methods, loaded=True)
    return share_network