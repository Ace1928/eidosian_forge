import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
@mock.patch.object(peer.Peer, 'is_four_octet_as_number_cap_valid', mock.MagicMock(return_value=False))
def test_trans_as_path_no_trans(self):
    input_as_path = [[65000, 4000, 40000, 30000, 40001]]
    expected_as_path = [[65000, 4000, 40000, 30000, 40001]]
    expected_as4_path = None
    self._test_trans_as_path(input_as_path, expected_as_path, expected_as4_path)