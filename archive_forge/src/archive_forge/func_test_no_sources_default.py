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
def test_no_sources_default(self):
    with base.mock.patch.object(self.conf, '_open_source_from_opt_group') as open_source:
        open_source.side_effect = AssertionError('should not be called')
        self.conf([])