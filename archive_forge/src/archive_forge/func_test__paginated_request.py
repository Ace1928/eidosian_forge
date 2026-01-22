import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
def test__paginated_request(self):
    DigitalOceanMockHttp.type = 'page_1'
    actions = self.driver._paginated_request('/v2/actions', 'actions')
    self.assertEqual(actions[0]['id'], 12345671)
    self.assertEqual(actions[0]['status'], 'completed')