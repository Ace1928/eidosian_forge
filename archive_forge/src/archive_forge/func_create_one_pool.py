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
def create_one_pool(attrs=None):
    """Create a fake pool.

    :param dict attrs:
        A dictionary with all attributes of the pool
    :return:
        A FakeResource object with pool name and attrs.
    """
    pool_info = {'name': 'host@lvmdriver-1#lvmdriver-1', 'capabilities': {'storage_protocol': 'iSCSI', 'thick_provisioning_support': False, 'thin_provisioning_support': True, 'total_volumes': 99, 'total_capacity_gb': 1000.0, 'allocated_capacity_gb': 100, 'max_over_subscription_ratio': 200.0}}
    pool_info.update(attrs or {})
    pool = _stats.Pools(**pool_info)
    return pool