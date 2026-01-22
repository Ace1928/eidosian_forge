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
def create_one_type_access(attrs=None):
    """Create a fake volume type access for project.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A FakeResource object, with  Volume_type_ID and Project_ID.
    """
    if attrs is None:
        attrs = {}
    type_access_attrs = {'volume_type_id': 'volume-type-id-' + uuid.uuid4().hex, 'project_id': 'project-id-' + uuid.uuid4().hex}
    type_access_attrs.update(attrs)
    type_access = fakes.FakeResource(None, type_access_attrs, loaded=True)
    return type_access