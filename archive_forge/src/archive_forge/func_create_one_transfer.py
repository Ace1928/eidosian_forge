import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_transfer(attrs=None, methods=None):
    """Create a fake share transfer

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    now_time = datetime.datetime.now()
    delta_time = now_time + datetime.timedelta(minutes=5)
    share_transfer = {'accepted': 'False', 'auth_key': 'auth-key-' + uuid.uuid4().hex, 'created_at': now_time.isoformat(), 'destination_project_id': None, 'expires_at': delta_time.isoformat(), 'id': 'transfer-id-' + uuid.uuid4().hex, 'name': 'name-' + uuid.uuid4().hex, 'resource_id': 'resource-id-' + uuid.uuid4().hex, 'resource_type': 'share', 'source_project_id': 'source-project-id-' + uuid.uuid4().hex}
    share_transfer.update(attrs)
    share_transfer = osc_fakes.FakeResource(info=copy.deepcopy(share_transfer), methods=methods, loaded=True)
    return share_transfer