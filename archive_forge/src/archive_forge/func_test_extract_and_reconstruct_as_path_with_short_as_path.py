import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
@mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
def test_extract_and_reconstruct_as_path_with_short_as_path(self):
    in_as_path_value = [[1000, 23456, 3000]]
    in_as4_path_value = [[2000, 3000, 4000, 5000]]
    in_aggregator_as_number = 4000
    in_aggregator_addr = '10.0.0.1'
    ex_as_path_value = [[1000, 23456, 3000]]
    ex_aggregator_as_number = 4000
    ex_aggregator_addr = '10.0.0.1'
    path_attributes = [bgp.BGPPathAttributeAsPath(value=in_as_path_value), bgp.BGPPathAttributeAs4Path(value=in_as4_path_value), bgp.BGPPathAttributeAggregator(as_number=in_aggregator_as_number, addr=in_aggregator_addr)]
    self._test_extract_and_reconstruct_as_path(path_attributes, ex_as_path_value, ex_aggregator_as_number, ex_aggregator_addr)