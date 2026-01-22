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
def import_backup_record():
    """Creates a fake backup record import response from a backup.

    :return: The fake backup object that was encoded.
    """
    return {'backup': {'id': 'backup.id', 'name': 'backup.name', 'links': [{'href': 'link1', 'rel': 'self'}, {'href': 'link2', 'rel': 'bookmark'}]}}