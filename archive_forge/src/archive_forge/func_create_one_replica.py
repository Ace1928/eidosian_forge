import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_replica(attrs=None, methods=None):
    """Create a fake share replica

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_replica = {'availability_zone': None, 'cast_rules_to_readonly': True, 'created_at': datetime.datetime.now().isoformat(), 'host': None, 'id': 'replica-id-' + uuid.uuid4().hex, 'replica_state': None, 'share_id': 'share-id-' + uuid.uuid4().hex, 'share_network_id': None, 'share_server_id': None, 'status': None, 'updated_at': None}
    share_replica.update(attrs)
    share_replica = osc_fakes.FakeResource(info=copy.deepcopy(share_replica), methods=methods, loaded=True)
    return share_replica