import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_sharetype(attrs=None, methods=None):
    """Create a fake share type

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_type_info = {'required_extra_specs': {'driver_handles_share_servers': True}, 'share_type_access:is_public': True, 'extra_specs': {'replication_type': 'readable', 'driver_handles_share_servers': True, 'mount_snapshot_support': False, 'revert_to_snapshot_support': False, 'create_share_from_snapshot_support': True, 'snapshot_support': True}, 'id': 'share-type-id-' + uuid.uuid4().hex, 'name': 'share-type-name-' + uuid.uuid4().hex, 'is_default': False, 'description': 'share-type-description-' + uuid.uuid4().hex}
    share_type_info.update(attrs)
    share_type = osc_fakes.FakeResource(info=copy.deepcopy(share_type_info), methods=methods, loaded=True)
    return share_type