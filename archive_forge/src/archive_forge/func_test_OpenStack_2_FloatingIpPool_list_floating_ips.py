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
def test_OpenStack_2_FloatingIpPool_list_floating_ips(self):
    pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
    ret = pool.list_floating_ips()
    self.assertEqual(ret[0].id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
    self.assertEqual(ret[0].get_pool(), pool)
    self.assertEqual(ret[0].ip_address, '10.3.1.42')
    self.assertEqual(ret[0].get_node_id(), None)
    self.assertEqual(ret[1].id, '04c5336a-0629-4694-ba30-04b0bdfa88a4')
    self.assertEqual(ret[1].get_pool(), pool)
    self.assertEqual(ret[1].ip_address, '10.3.1.1')
    self.assertEqual(ret[1].get_node_id(), 'fcfc96da-19e2-40fd-8497-f29da1b21143')
    self.assertEqual(ret[2].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
    self.assertEqual(ret[2].get_pool(), pool)
    self.assertEqual(ret[2].ip_address, '10.3.1.2')
    self.assertEqual(ret[2].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')
    self.assertEqual(ret[3].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
    self.assertEqual(ret[3].get_pool(), pool)
    self.assertEqual(ret[3].ip_address, '10.3.1.3')
    self.assertEqual(ret[3].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')