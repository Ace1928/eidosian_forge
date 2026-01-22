import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_format_parameters_multiple_values_per_pamaters(self):
    p = utils.format_parameters(['status=COMPLETE', 'status=FAILED'])
    self.assertIn('status', p)
    self.assertIn('COMPLETE', p['status'])
    self.assertIn('FAILED', p['status'])