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
def test_unauthorized_clears_cached_auth_context(self):
    auth_cache = OpenStackMockAuthCache()
    self.assertEqual(len(auth_cache), 0)
    kwargs = self.driver_kwargs.copy()
    kwargs['ex_auth_cache'] = auth_cache
    driver = self.driver_type(*self.driver_args, **kwargs)
    driver.list_nodes()
    self.assertEqual(len(auth_cache), 1)
    self.driver_klass.connectionCls.conn_class.type = 'UNAUTHORIZED'
    with pytest.raises(BaseHTTPError):
        driver.list_nodes()
    self.assertEqual(len(auth_cache), 0)