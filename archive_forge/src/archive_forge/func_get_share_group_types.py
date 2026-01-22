import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def get_share_group_types(share_group_types=None, count=2):
    """Get an iterable MagicMock object with a list of faked group types.

        If types list is provided, then initialize the Mock object with the
        list. Otherwise create one.

        :param List types:
            A list of FakeResource objects faking types
        :param Integer count:
            The number of group types to be faked
        :return
            An iterable Mock object with side_effect set to a list of faked
            group types
        """
    if share_group_types is None:
        share_group_types = FakeShareGroupType.share_group_types(count)
    return mock.Mock(side_effect=share_group_types)