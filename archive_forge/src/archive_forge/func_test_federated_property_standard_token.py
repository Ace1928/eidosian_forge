import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_federated_property_standard_token(self):
    """Check if is_federated property returns expected value."""
    token = fixture.V3Token()
    token.set_project_scope()
    auth_ref = access.create(body=token)
    self.assertFalse(auth_ref.is_federated)