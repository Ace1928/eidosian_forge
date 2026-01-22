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
def test_ex_update_image(self):
    image_id = 'f24a3c1b-d52a-4116-91da-25b3eee8f55e'
    data = {'op': 'replace', 'path': '/visibility', 'value': 'shared'}
    image = self.driver.ex_update_image(image_id, data)
    self.assertEqual(image.name, 'hypernode')
    self.assertIsNone(image.extra['serverId'])
    self.assertEqual(image.extra['minDisk'], 40)
    self.assertEqual(image.extra['minRam'], 0)
    self.assertEqual(image.extra['visibility'], 'shared')