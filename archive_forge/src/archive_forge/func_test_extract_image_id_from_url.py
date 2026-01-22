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
def test_extract_image_id_from_url(self):
    url = 'http://127.0.0.1/v1.1/68/images/1d4a8ea9-aae7-4242-a42d-5ff4702f2f14'
    url_two = 'http://127.0.0.1/v1.1/68/images/13'
    image_id = self.driver._extract_image_id_from_url(url)
    image_id_two = self.driver._extract_image_id_from_url(url_two)
    self.assertEqual(image_id, '1d4a8ea9-aae7-4242-a42d-5ff4702f2f14')
    self.assertEqual(image_id_two, '13')