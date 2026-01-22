import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_yaml_formatter(self):
    self.assertEqual('null\n...\n', utils.yaml_formatter(None))
    self.assertEqual('{}\n', utils.yaml_formatter({}))
    self.assertEqual('foo: bar\n', utils.yaml_formatter({'foo': 'bar'}))