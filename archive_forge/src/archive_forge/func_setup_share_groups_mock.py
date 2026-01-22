import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def setup_share_groups_mock(self):
    self.share_group_mock = self.app.client_manager.share.share_groups
    self.share_group_mock.reset_mock()
    share_group = manila_fakes.FakeShareGroup.create_one_share_group()
    self.share_group_mock.get = mock.Mock(return_value=share_group)
    return share_group