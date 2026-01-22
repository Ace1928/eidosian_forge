import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import endpoints
def test_list_invalid_interface(self):
    interface = uuid.uuid4().hex
    expected_path = 'v3/%s?interface=%s' % (self.collection_key, interface)
    self.assertRaises(exceptions.ValidationError, self.manager.list, expected_path=expected_path, interface=interface)