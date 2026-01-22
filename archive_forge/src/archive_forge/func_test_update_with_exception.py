import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_update_with_exception(self):
    share_nw = 'fake share nw'
    self.assertRaises(exceptions.CommandError, self.manager.update, share_nw)