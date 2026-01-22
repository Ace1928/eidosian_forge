import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_project_domain(self):
    self.assertRaises(exceptions.OptionError, self.create, username=uuid.uuid4().hex, passcode=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex, project_name=uuid.uuid4().hex)