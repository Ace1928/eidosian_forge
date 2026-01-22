import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameter_file_none(self):
    self.assertEqual({}, utils.format_parameter_file(None))