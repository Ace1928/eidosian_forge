from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_list_flavors_is_public_false(self):
    fl = self.cs.flavors.list(is_public=False)
    self.assert_request_id(fl, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/flavors/detail?is_public=False')
    for flavor in fl:
        self.assertIsInstance(flavor, self.flavor_type)