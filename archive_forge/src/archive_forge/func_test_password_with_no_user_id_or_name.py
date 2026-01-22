import copy
import json
import time
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v2
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_password_with_no_user_id_or_name(self):
    self.assertRaises(TypeError, v2.Password, self.TEST_URL, password=self.TEST_PASS)