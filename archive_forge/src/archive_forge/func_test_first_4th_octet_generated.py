import itertools
import random
import socket
from unittest import mock
from neutron_lib import constants
from neutron_lib.tests import _base as base
from neutron_lib.utils import net
@mock.patch.object(random, 'getrandbits', return_value=162)
def test_first_4th_octet_generated(self, mock_rnd):
    mac = net.get_random_mac(['aa', 'bb', 'cc', '00', 'ee', 'ff'])
    self.assertEqual('aa:bb:cc:a2:a2:a2', mac)
    mock_rnd.assert_called_with(8)