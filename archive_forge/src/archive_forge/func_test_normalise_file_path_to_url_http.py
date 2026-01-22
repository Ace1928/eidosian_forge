import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_normalise_file_path_to_url_http(self):
    self.assertEqual('http://localhost/foo', utils.normalise_file_path_to_url('http://localhost/foo'))