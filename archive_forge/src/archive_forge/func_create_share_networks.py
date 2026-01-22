import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_networks(attrs=None, count=2):
    """Create multiple fake share networks.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share networks to be faked

        :return:
            A list of FakeResource objects
        """
    share_networks = []
    for n in range(count):
        share_networks.append(FakeShareNetwork.create_one_share_network(attrs))
    return share_networks