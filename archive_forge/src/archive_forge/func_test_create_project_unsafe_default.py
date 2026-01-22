import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_create_project_unsafe_default(self):
    """Check default for unsafe names for ``POST /projects``."""
    unsafe_name = 'i am not / safe'
    ref = unit.new_project_ref(name=unsafe_name)
    self.post('/projects', body={'project': ref})