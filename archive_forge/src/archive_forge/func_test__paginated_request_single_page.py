import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import DIGITALOCEAN_v1_PARAMS, DIGITALOCEAN_v2_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.digitalocean import DigitalOcean_v1_Error
from libcloud.compute.drivers.digitalocean import DigitalOceanNodeDriver
def test__paginated_request_single_page(self):
    nodes = self.driver._paginated_request('/v2/droplets', 'droplets')
    self.assertEqual(nodes[0]['name'], 'ubuntu-s-1vcpu-1gb-sfo3-01')
    self.assertEqual(nodes[0]['image']['id'], 69463186)
    self.assertEqual(nodes[0]['size_slug'], 's-1vcpu-1gb')