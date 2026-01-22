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
def test_build_xmltree_description(self):
    instantiate_xml = Instantiate_1_5_VAppXML(name='testNode', template='https://vm-vcloud/api/vAppTemplate/vappTemplate-ac1bc027-bf8c-4050-8643-4971f691c158', network=None, vm_network=None, vm_fence=None, description=None)
    self.assertIsNone(instantiate_xml.description)
    self.assertIsNone(instantiate_xml.root.find('Description'))
    test_description = 'Test Description'
    instantiate_xml = Instantiate_1_5_VAppXML(name='testNode', template='https://vm-vcloud/api/vAppTemplate/vappTemplate-ac1bc027-bf8c-4050-8643-4971f691c158', network=None, vm_network=None, vm_fence=None, description=test_description)
    self.assertEqual(instantiate_xml.description, test_description)
    description_elem = instantiate_xml.root.find('Description')
    self.assertIsNotNone(description_elem)
    self.assertEqual(description_elem.text, test_description)