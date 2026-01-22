import itertools
import random
import socket
from unittest import mock
from neutron_lib import constants
from neutron_lib.tests import _base as base
from neutron_lib.utils import net
@mock.patch.object(random, 'getrandbits', return_value=162)
def test_short_supplied_mac(self, mock_rnd):
    mac_base = '12:34:56:78'
    mac = mac_base.split(':')
    generator = itertools.islice(net.random_mac_generator(mac), 1)
    self.assertEqual(['12:34:56:78:a2:a2'], list(generator))
    mock_rnd.assert_called_with(8)