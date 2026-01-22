import random
from unittest import mock
import uuid
from cinderclient import api_versions
from openstack.block_storage.v3 import _proxy
from openstack.block_storage.v3 import availability_zone as _availability_zone
from openstack.block_storage.v3 import extension as _extension
from openstack.block_storage.v3 import resource_filter as _filters
from openstack.block_storage.v3 import volume as _volume
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_v2_fakes
def create_one_manage_record(attrs=None, snapshot=False):
    manage_dict = {'reference': {'source-name': 'fake-volume'}, 'size': '1', 'safe_to_manage': False, 'reason_not_safe': 'already managed', 'cinder_id': 'fake-volume', 'extra_info': None}
    if snapshot:
        manage_dict['source_reference'] = {'source-name': 'fake-source'}
    attrs = attrs or {}
    manage_dict.update(attrs)
    manage_record = fakes.FakeResource(None, manage_dict, loaded=True)
    return manage_record