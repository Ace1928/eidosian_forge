import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_share(attrs=None, methods=None):
    """Create a fake share.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with flavor_id, image_id, and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_info = {'status': None, 'share_server_id': None, 'project_id': 'project-id-' + uuid.uuid4().hex, 'name': 'share-name-' + uuid.uuid4().hex, 'share_type': 'share-type-' + uuid.uuid4().hex, 'share_type_name': 'default', 'availability_zone': None, 'created_at': 'time-' + uuid.uuid4().hex, 'share_network_id': None, 'share_group_id': None, 'share_proto': 'NFS', 'host': None, 'access_rules_status': 'active', 'has_replicas': False, 'replication_type': None, 'task_state': None, 'snapshot_support': True, 'snapshot_id': None, 'is_public': True, 'metadata': {}, 'id': 'share-id-' + uuid.uuid4().hex, 'size': random.randint(1, 20), 'description': 'share-description-' + uuid.uuid4().hex, 'user_id': 'share-user-id-' + uuid.uuid4().hex, 'create_share_from_snapshot_support': False, 'mount_snapshot_support': False, 'revert_to_snapshot_support': False, 'source_share_group_snapshot_member_id': None, 'scheduler_hints': {}}
    share_info.update(attrs)
    share = osc_fakes.FakeResource(info=copy.deepcopy(share_info), methods=methods, loaded=True)
    return share