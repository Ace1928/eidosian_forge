import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import endpoints
def test_update_admin_interface(self):
    ref = self.new_ref(interface='admin')
    self.test_update(ref)