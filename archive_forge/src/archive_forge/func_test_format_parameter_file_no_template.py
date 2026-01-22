import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameter_file_no_template(self):
    tmpl_file = None
    contents = 'DBUsername=wp\nDBPassword=verybadpassword'
    utils.read_url_content = mock.MagicMock()
    utils.read_url_content.return_value = 'DBUsername=wp\nDBPassword=verybadpassword'
    p = utils.format_parameter_file(['env_file1=test_file1'], tmpl_file)
    self.assertEqual({'env_file1': contents}, p)