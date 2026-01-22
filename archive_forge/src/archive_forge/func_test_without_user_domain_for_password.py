import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_user_domain_for_password(self):
    self.assertRaises(exceptions.OptionError, self.create, methods=['v3password'], username=uuid.uuid4().hex, project_name=uuid.uuid4().hex, project_domain_id=uuid.uuid4().hex)