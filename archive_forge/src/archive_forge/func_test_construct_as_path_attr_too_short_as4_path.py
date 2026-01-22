import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
def test_construct_as_path_attr_too_short_as4_path(self):
    input_as_path = [[65000, 4000, 23456, 23456, 40001]]
    input_as4_path = [[300000, 40001]]
    expected_as_path = [[65000, 4000, 23456, 300000, 40001]]
    self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)