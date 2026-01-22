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
def test_use_env_false_allows_files(self):
    file_value = 'hello'
    env_value = 'goodbye'
    os.environ['OS_FOO__BAR'] = env_value
    self.conf(args=[], use_env=False)
    self.conf_fixture.load_raw_values(group='foo', bar=file_value)
    self.assertEqual(file_value, self.conf['foo']['bar'])
    self.conf.reset()
    self.conf(args=[], use_env=True)
    self.assertEqual(env_value, self.conf['foo']['bar'])