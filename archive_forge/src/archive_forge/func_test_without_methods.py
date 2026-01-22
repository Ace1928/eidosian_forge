import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_methods(self):
    self.assertRaises(exceptions.OptionError, self.create, username=uuid.uuid4().hex, passcode=uuid.uuid4().hex)