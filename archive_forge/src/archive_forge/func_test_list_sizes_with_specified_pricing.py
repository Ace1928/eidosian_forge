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
def test_list_sizes_with_specified_pricing(self):
    pricing = {str(i): i * 5.0 for i in range(1, 9)}
    set_pricing(driver_type='compute', driver_name=self.driver.api_name, pricing=pricing)
    sizes = self.driver.list_sizes()
    self.assertEqual(len(sizes), 8, 'Wrong sizes count')
    for size in sizes:
        self.assertTrue(isinstance(size.price, float), 'Wrong size price type')
        self.assertEqual(size.price, pricing[size.id], 'Size price should match')