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
def create_volume_types(attrs=None, count=2):
    """Create multiple fake volume_types.

    :param dict attrs:
        A dictionary with all attributes
    :param int count:
        The number of types to fake
    :return:
        A list of FakeResource objects faking the types
    """
    volume_types = []
    for i in range(0, count):
        volume_type = create_one_volume_type(attrs)
        volume_types.append(volume_type)
    return volume_types