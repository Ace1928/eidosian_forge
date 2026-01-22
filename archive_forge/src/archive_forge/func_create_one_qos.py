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
def create_one_qos(attrs=None):
    """Create a fake Qos specification.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A FakeResource object with id, name, consumer, etc.
    """
    attrs = attrs or {}
    qos_info = {'id': 'qos-id-' + uuid.uuid4().hex, 'name': 'qos-name-' + uuid.uuid4().hex, 'consumer': 'front-end', 'specs': {'foo': 'bar', 'iops': '9001'}}
    qos_info.update(attrs)
    qos = fakes.FakeResource(info=copy.deepcopy(qos_info), loaded=True)
    return qos