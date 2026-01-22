from unittest import mock
from oslo_serialization import jsonutils
import testtools
from heatclient.tests.unit import fakes
from heatclient.v1 import build_info
def test_build_info_returns_the_response_body(self):
    response = self.manager.build_info()
    self.assertEqual('body', response)