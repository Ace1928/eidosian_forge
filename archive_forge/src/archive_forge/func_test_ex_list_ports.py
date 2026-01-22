import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
def test_ex_list_ports(self):
    ports = self.driver.ex_list_ports()
    port = ports[0]
    self.assertEqual(port.id, '126da55e-cfcb-41c8-ae39-a26cb8a7e723')
    self.assertEqual(port.state, OpenStack_2_PortInterfaceState.BUILD)
    self.assertEqual(port.created, '2018-07-04T14:38:18Z')
    self.assertEqual(port.extra['network_id'], '123c8a8c-6427-4e8f-a805-2035365f4d43')
    self.assertEqual(port.extra['project_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
    self.assertEqual(port.extra['tenant_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
    self.assertEqual(port.extra['name'], '')