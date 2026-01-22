import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_lock(attrs=None, methods=None):
    """Create a fake resource lock

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    now_time = datetime.datetime.now()
    delta_time = now_time + datetime.timedelta(minutes=5)
    lock = {'id': str(uuid.uuid4()), 'resource_id': str(uuid.uuid4()), 'resource_type': 'share', 'resource_action': 'delete', 'created_at': now_time.isoformat(), 'updated_at': delta_time.isoformat(), 'project_id': uuid.uuid4().hex, 'user_id': uuid.uuid4().hex, 'lock_context': 'user', 'lock_reason': 'created by func tests'}
    lock.update(attrs)
    lock = osc_fakes.FakeResource(info=copy.deepcopy(lock), methods=methods, loaded=True)
    return lock