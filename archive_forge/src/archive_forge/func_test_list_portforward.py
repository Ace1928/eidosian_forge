import sys
import json
import time
import base64
import unittest
from unittest import mock
import libcloud.common.gig_g8
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, StorageVolume
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gig_g8 import G8Network, G8NodeDriver, G8PortForward
def test_list_portforward(self):
    network = self.driver.ex_list_networks()[0]
    forwards = self.driver.ex_list_portforwards(network)
    self.assertGreater(len(forwards), 0)
    for forward in forwards:
        self.assertIsInstance(forward, G8PortForward)