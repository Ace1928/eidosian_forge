from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
def test_api_version_is_gte_27(self):
    api_version = api_versions.APIVersion('2.27')
    headers = {}
    api_versions.update_headers(headers, api_version)
    self.assertIn('X-OpenStack-Nova-API-Version', headers)
    self.assertIn('OpenStack-API-Version', headers)
    self.assertEqual(api_version.get_string(), headers['X-OpenStack-Nova-API-Version'])
    self.assertEqual('%s %s' % (api_versions.SERVICE_TYPE, api_version.get_string()), headers['OpenStack-API-Version'])