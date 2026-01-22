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
def create_service_log_level_entry(attrs=None):
    service_log_level_info = {'host': 'host_test', 'binary': 'cinder-api', 'prefix': 'cinder.api.common', 'level': 'DEBUG'}
    attrs = attrs or {}
    service_log_level_info.update(attrs)
    service_log_level = fakes.FakeResource(None, service_log_level_info, loaded=True)
    return service_log_level