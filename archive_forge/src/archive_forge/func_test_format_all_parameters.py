import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_all_parameters(self):
    tmpl_file = '/opt/stack/template.yaml'
    contents = 'DBUsername=wp\nDBPassword=verybadpassword'
    params = ['KeyName=heat_key;UpstreamDNS=8.8.8.8']
    utils.read_url_content = mock.MagicMock()
    utils.read_url_content.return_value = 'DBUsername=wp\nDBPassword=verybadpassword'
    p = utils.format_all_parameters(params, ['env_file1=test_file1'], template_file=tmpl_file)
    self.assertEqual({'KeyName': 'heat_key', 'UpstreamDNS': '8.8.8.8', 'env_file1': contents}, p)