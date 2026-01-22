import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameters_split(self):
    p = utils.format_parameters(['KeyName=heat_key;DnsSecKey=hsgx1m31PbamNF4WEcHlwjIlCGgifOdoB58/wwC7a4oAONQ/fDV5ctqrYBoLlKHhTfkyQEw9iVScKYZbbMtMNg==;UpstreamDNS=8.8.8.8'])
    self.assertEqual({'KeyName': 'heat_key', 'DnsSecKey': 'hsgx1m31PbamNF4WEcHlwjIlCGgifOdoB58/wwC7a4oAONQ/fDV5ctqrYBoLlKHhTfkyQEw9iVScKYZbbMtMNg==', 'UpstreamDNS': '8.8.8.8'}, p)