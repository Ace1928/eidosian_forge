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
def test_auth_host_passed(self):
    forced_auth = 'http://x.y.z.y:5000'
    d = OpenStack_1_0_NodeDriver('user', 'correct_password', ex_force_auth_version='2.0_password', ex_force_auth_url='http://x.y.z.y:5000', ex_tenant_name='admin')
    self.assertEqual(d._ex_force_auth_url, forced_auth)
    with requests_mock.Mocker() as mock:
        body2 = ComputeFileFixtures('openstack').load('_v2_0__auth.json')
        mock.register_uri('POST', 'http://x.y.z.y:5000/v2.0/tokens', text=body2, headers={'content-type': 'application/json; charset=UTF-8'})
        d.connection._populate_hosts_and_request_paths()
        self.assertEqual(d.connection.host, 'test_endpoint.com')