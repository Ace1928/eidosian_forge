import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_groups(attrs=None, count=2):
    """Create multiple fake groups.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share groups to be faked
        :return:
            A list of FakeResource objects
        """
    share_groups = []
    for n in range(0, count):
        share_groups.append(FakeShareGroup.create_one_share_group(attrs))
    return share_groups