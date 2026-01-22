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
def create_one_default_vol_quota(attrs=None):
    """Create one quota"""
    attrs = attrs or {}
    quota_attrs = {'id': 'project-id-' + uuid.uuid4().hex, 'backups': 100, 'backup_gigabytes': 100, 'gigabytes': 100, 'per_volume_gigabytes': 100, 'snapshots': 100, 'volumes': 100}
    quota_attrs.update(attrs)
    quota = fakes.FakeResource(info=copy.deepcopy(quota_attrs), loaded=True)
    quota.project_id = quota_attrs['id']
    return quota