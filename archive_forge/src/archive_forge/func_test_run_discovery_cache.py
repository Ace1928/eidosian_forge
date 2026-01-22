import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
@mock.patch('keystoneauth1.discover.get_discovery')
@mock.patch('keystoneauth1.discover.EndpointData._get_discovery_url_choices')
def test_run_discovery_cache(self, mock_url_choices, mock_get_disc):
    mock_get_disc.side_effect = exceptions.DiscoveryFailure()
    mock_url_choices.return_value = ('url1', 'url2', 'url1', 'url3')
    epd = discover.EndpointData()
    epd._run_discovery(session='sess', cache='cache', min_version='min', max_version='max', project_id='projid', allow_version_hack='allow_hack', discover_versions='disc_vers')
    self.assertEqual(3, mock_get_disc.call_count)
    mock_get_disc.assert_has_calls([mock.call('sess', url, cache='cache', authenticated=False) for url in ('url1', 'url2', 'url3')])