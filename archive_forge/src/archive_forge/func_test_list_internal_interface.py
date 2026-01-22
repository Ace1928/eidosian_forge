import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import endpoints
def test_list_internal_interface(self):
    interface = 'admin'
    expected_path = 'v3/%s?interface=%s' % (self.collection_key, interface)
    self.test_list(expected_path=expected_path, interface=interface)