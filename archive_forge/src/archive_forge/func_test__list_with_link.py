from unittest import mock
import requests
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import client
from cinderclient.v3 import volumes
def test__list_with_link(self):
    api = mock.Mock()
    api.client.get.side_effect = [(mock.sentinel.resp, {'resp_keys': [{'name': '1'}], 'resp_keys_links': [{'rel': 'next', 'href': mock.sentinel.u2}]}), (mock.sentinel.resp, {'resp_keys': [{'name': '2'}], 'resp_keys_links': [{'rel': 'next', 'href': mock.sentinel.u3}]}), (mock.sentinel.resp, {'resp_keys': [{'name': '3'}], 'resp_keys_links': [{'rel': 'next', 'href': None}]})]
    manager = test_utils.FakeManager(api)
    res = manager._list(mock.sentinel.url, 'resp_keys')
    api.client.get.assert_has_calls([mock.call(mock.sentinel.url), mock.call(mock.sentinel.u2), mock.call(mock.sentinel.u3)])
    result = [r.name for r in res]
    self.assertListEqual(['1', '2', '3'], result)