import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_export_locations(attrs=None, count=2):
    """Create multiple fake export locations.

        :param Dictionary attrs:
            A dictionary with all attributes

        :param Integer count:
            The number of share export locations to be faked

        :return:
            A list of FakeResource objects
        """
    share_export_locations = []
    for n in range(0, count):
        share_export_locations.append(FakeShareExportLocation.create_one_export_location(attrs))
    return share_export_locations