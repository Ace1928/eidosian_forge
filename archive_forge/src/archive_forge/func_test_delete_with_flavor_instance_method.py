from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_delete_with_flavor_instance_method(self):
    f = self.cs.flavors.get(2)
    ret = f.delete()
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('DELETE', '/flavors/2')