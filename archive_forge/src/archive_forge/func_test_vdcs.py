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
def test_vdcs(self):
    vdcs = self.driver.vdcs
    self.assertEqual(len(vdcs), 1)
    self.assertEqual(vdcs[0].id, 'https://vm-vcloud/api/vdc/3d9ae28c-1de9-4307-8107-9356ff8ba6d0')
    self.assertEqual(vdcs[0].name, 'MyVdc')
    self.assertEqual(vdcs[0].allocation_model, 'AllocationPool')
    self.assertEqual(vdcs[0].storage.limit, 5120000)
    self.assertEqual(vdcs[0].storage.used, 1984512)
    self.assertEqual(vdcs[0].storage.units, 'MB')
    self.assertEqual(vdcs[0].cpu.limit, 160000)
    self.assertEqual(vdcs[0].cpu.used, 0)
    self.assertEqual(vdcs[0].cpu.units, 'MHz')
    self.assertEqual(vdcs[0].memory.limit, 527360)
    self.assertEqual(vdcs[0].memory.used, 130752)
    self.assertEqual(vdcs[0].memory.units, 'MB')