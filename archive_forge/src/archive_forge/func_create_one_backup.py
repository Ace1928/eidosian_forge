import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_backup(attrs=None, methods=None):
    """Create a fake share backup

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_backup = {'id': 'backup-id-' + uuid.uuid4().hex, 'share_id': 'share-id-' + uuid.uuid4().hex, 'status': None, 'name': None, 'description': None, 'size': '0', 'created_at': datetime.datetime.now().isoformat(), 'updated_at': datetime.datetime.now().isoformat(), 'availability_zone': None, 'progress': None, 'restore_progress': None, 'host': None, 'topic': None}
    share_backup.update(attrs)
    share_backup = osc_fakes.FakeResource(info=copy.deepcopy(share_backup), methods=methods, loaded=True)
    return share_backup