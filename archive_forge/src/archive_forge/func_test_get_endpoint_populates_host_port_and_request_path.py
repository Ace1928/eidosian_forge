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
def test_get_endpoint_populates_host_port_and_request_path(self):
    self.driver.connection.get_endpoint = lambda: 'http://endpoint_auth_url.com:1555/service_url'
    self.driver.connection.auth_token = None
    self.driver.connection._ex_force_base_url = None
    self.driver.connection._populate_hosts_and_request_paths()
    self.assertEqual(self.driver.connection.host, 'endpoint_auth_url.com')
    self.assertEqual(self.driver.connection.port, 1555)
    self.assertEqual(self.driver.connection.request_path, '/service_url')