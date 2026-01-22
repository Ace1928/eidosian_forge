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
def get_consistency_groups(consistency_groups=None, count=2):
    """Note:

    Get an iterable MagicMock object with a list of faked
    consistency_groups.

    If consistency_groups list is provided, then initialize
    the Mock object with the list. Otherwise create one.

    :param List consistency_groups:
        A list of FakeResource objects faking consistency_groups
    :param Integer count:
        The number of consistency_groups to be faked
    :return
        An iterable Mock object with side_effect set to a list of faked
        consistency_groups
    """
    if consistency_groups is None:
        consistency_groups = create_consistency_groups(count)
    return mock.Mock(side_effect=consistency_groups)