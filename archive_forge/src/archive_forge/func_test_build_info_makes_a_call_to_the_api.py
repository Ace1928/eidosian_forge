from unittest import mock
from oslo_serialization import jsonutils
import testtools
from heatclient.tests.unit import fakes
from heatclient.v1 import build_info
def test_build_info_makes_a_call_to_the_api(self):
    self.manager.build_info()
    self.client.get.assert_called_once_with('/build_info')