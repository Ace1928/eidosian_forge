import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
def test_ex_list_bgp_sessions_for_project(self):
    sessions = self.driver.ex_list_bgp_sessions_for_project(ex_project_id='4b653fce-6405-4300-9f7d-c587b7888fe5')
    self.assertEqual(sessions['bgp_sessions'][0]['status'], 'down')