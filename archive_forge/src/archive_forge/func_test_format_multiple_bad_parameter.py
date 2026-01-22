import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_multiple_bad_parameter(self):
    params = ['KeyName=heat_key', 'UpstreamDNS8.8.8.8']
    ex = self.assertRaises(exc.CommandError, utils.format_parameters, params)
    self.assertEqual('Malformed parameter(UpstreamDNS8.8.8.8). Use the key=value format.', str(ex))