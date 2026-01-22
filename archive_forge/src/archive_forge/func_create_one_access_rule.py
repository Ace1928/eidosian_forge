import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_access_rule(attrs={}):
    """Create a fake snapshot access rule

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    snapshot_access_rule = {'access_to': 'demo', 'access_type': 'user', 'id': 'access_rule-id-' + uuid.uuid4().hex, 'state': 'queued_to_apply'}
    snapshot_access_rule.update(attrs)
    snapshot_access_rule = osc_fakes.FakeResource(info=copy.deepcopy(snapshot_access_rule), loaded=True)
    return snapshot_access_rule