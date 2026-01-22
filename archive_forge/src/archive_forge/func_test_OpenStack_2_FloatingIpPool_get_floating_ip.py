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
def test_OpenStack_2_FloatingIpPool_get_floating_ip(self):
    pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
    ret = pool.get_floating_ip('10.3.1.42')
    self.assertEqual(ret.id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
    self.assertEqual(ret.pool, pool)
    self.assertEqual(ret.ip_address, '10.3.1.42')
    self.assertEqual(ret.node_id, None)