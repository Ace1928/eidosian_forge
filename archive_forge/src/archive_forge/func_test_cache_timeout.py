import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_utils.timeutils.utcnow_ts')
def test_cache_timeout(self, mock_utcnow_ts):
    mock_utcnow_ts.side_effect = [100, 125, 150, 175, 195, 200, 225]
    cache = service.MemoryCache()
    cache.put('key1', 'value1', 10)
    cache.put('key2', 'value2', 75)
    cache.put('key3', 'value3', 100)
    self.assertIsNone(cache.get('key1'))
    self.assertEqual('value2', cache.get('key2'))
    self.assertIsNone(cache.get('key2'))
    self.assertEqual('value3', cache.get('key3'))