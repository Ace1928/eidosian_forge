import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_share_group_snapshot(attrs=None, methods=None):
    """Create a fake share group snapshot

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_group_snapshot = {'status': 'available', 'share_group_id': 'share-group-id-' + uuid.uuid4().hex, 'name': None, 'created_at': datetime.datetime.now().isoformat(), 'project_id': 'project-id-' + uuid.uuid4().hex, 'id': 'share-group-snapshot-id-' + uuid.uuid4().hex, 'description': None}
    share_group_snapshot.update(attrs)
    share_group_snapshot = osc_fakes.FakeResource(info=copy.deepcopy(share_group_snapshot), methods=methods, loaded=True)
    return share_group_snapshot