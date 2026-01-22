import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
@requests_mock.mock()
def test_fetch_uri(self, m):
    m.get('https://bad.uri', status_code=404)
    self.assertRaises(HTTPError, _uri.URIConfigurationSource, 'https://bad.uri')
    m.get('https://good.uri', text='[DEFAULT]\nfoo=bar\n')
    source = _uri.URIConfigurationSource('https://good.uri')
    self.assertEqual('bar', source.get('DEFAULT', 'foo', cfg.StrOpt('foo'))[0])