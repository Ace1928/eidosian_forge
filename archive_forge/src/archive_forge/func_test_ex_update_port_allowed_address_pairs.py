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
def test_ex_update_port_allowed_address_pairs(self):
    allowed_address_pairs = [{'ip_address': '1.2.3.4'}, {'ip_address': '2.3.4.5'}]
    port = self.driver.ex_get_port('126da55e-cfcb-41c8-ae39-a26cb8a7e723')
    ret = self.driver.ex_update_port(port, allowed_address_pairs=allowed_address_pairs)
    self.assertEqual(ret.extra['allowed_address_pairs'], allowed_address_pairs)