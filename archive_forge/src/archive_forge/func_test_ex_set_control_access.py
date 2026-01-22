import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def test_ex_set_control_access(self):
    node = Node('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6b', 'testNode', NodeState.RUNNING, [], [], self.driver)
    control_access = ControlAccess(node, None, [Subject(name='MyGroup', type='group', access_level=ControlAccess.AccessLevel.FULL_CONTROL)])
    self.driver.ex_set_control_access(node, control_access)