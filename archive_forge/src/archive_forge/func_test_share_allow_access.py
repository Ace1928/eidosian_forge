from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_share_allow_access(self):
    access_level = 'fake_level'
    access_to = 'fake_credential'
    access_type = 'fake_type'
    self.share.allow(access_type, access_to, access_level)
    self.share.manager.allow.assert_called_once_with(self.share, access_type, access_to, access_level)