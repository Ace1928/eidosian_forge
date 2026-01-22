import copy
import random
from unittest import mock
import uuid
from cinderclient import api_versions
from openstack.block_storage.v2 import _proxy as block_storage_v2_proxy
from openstack.block_storage.v3 import backup as _backup
from openstack.block_storage.v3 import capabilities as _capabilities
from openstack.block_storage.v3 import stats as _stats
from openstack.block_storage.v3 import volume as _volume
from openstack.image.v2 import _proxy as image_v2_proxy
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def create_one_capability(attrs=None):
    """Create a fake volume backend capability.

    :param dict attrs:
        A dictionary with all attributes of the Capabilities.
    :return:
        A FakeResource object with capability name and attrs.
    """
    capability_info = {'namespace': 'OS::Storage::Capabilities::fake', 'vendor_name': 'OpenStack', 'volume_backend_name': 'lvmdriver-1', 'pool_name': 'pool', 'driver_version': '2.0.0', 'storage_protocol': 'iSCSI', 'display_name': 'Capabilities of Cinder LVM driver', 'description': 'Blah, blah.', 'visibility': 'public', 'replication_targets': [], 'properties': {'compression': {'title': 'Compression', 'description': 'Enables compression.', 'type': 'boolean'}, 'qos': {'title': 'QoS', 'description': 'Enables QoS.', 'type': 'boolean'}, 'replication': {'title': 'Replication', 'description': 'Enables replication.', 'type': 'boolean'}, 'thin_provisioning': {'title': 'Thin Provisioning', 'description': 'Sets thin provisioning.', 'type': 'boolean'}}}
    capability_info.update(attrs or {})
    capability = _capabilities.Capabilities(**capability_info)
    return capability