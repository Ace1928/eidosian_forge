import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_name_or_id(self):
    self.assertRaises(exceptions.OptionError, self.create, username=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex, application_credential_secret=uuid.uuid4().hex)