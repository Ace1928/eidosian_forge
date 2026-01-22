import copy
import random
from unittest import mock
import uuid
from novaclient import api_versions
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone as _availability_zone
from openstack.compute.v2 import extension as _extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstackclient.api import compute_v2
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def create_one_agent(attrs=None):
    """Create a fake agent.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A FakeResource object, with agent_id, os, and so on
    """
    attrs = attrs or {}
    agent_info = {'agent_id': 'agent-id-' + uuid.uuid4().hex, 'os': 'agent-os-' + uuid.uuid4().hex, 'architecture': 'agent-architecture', 'version': '8.0', 'url': 'http://127.0.0.1', 'md5hash': 'agent-md5hash', 'hypervisor': 'hypervisor'}
    agent_info.update(attrs)
    agent = fakes.FakeResource(info=copy.deepcopy(agent_info), loaded=True)
    return agent