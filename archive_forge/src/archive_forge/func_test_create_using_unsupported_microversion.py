from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_types as types
def test_create_using_unsupported_microversion(self):
    self.manager.api.api_version = manilaclient.API_MIN_VERSION
    self.assertRaises(exceptions.UnsupportedVersion, self.manager.create)