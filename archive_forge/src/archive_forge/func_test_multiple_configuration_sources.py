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
@base.mock.patch('oslo_config.sources._uri.URIConfigurationSource._fetch_uri', side_effect=opts_to_ini)
def test_multiple_configuration_sources(self, mock_fetch_uri):
    groups = ['ini_1', 'ini_2', 'ini_3']
    uri = make_uri('ini_3')
    for group in groups:
        self.conf_fixture.load_raw_values(group=group, driver='remote_file', uri=make_uri(group))
    self.conf_fixture.config(config_source=groups)
    self.conf._load_alternative_sources()
    self._register_opts(_extra_configs[uri]['data'])
    for option in _extra_configs[uri]['data']['DEFAULT']:
        self.assertEqual(option, self.conf[option])