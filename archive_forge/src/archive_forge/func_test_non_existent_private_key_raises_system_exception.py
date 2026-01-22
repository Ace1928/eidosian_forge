import os
import uuid
from keystone.common import jwt_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.token import provider
from keystone.token.providers import jws
def test_non_existent_private_key_raises_system_exception(self):
    private_key = os.path.join(CONF.jwt_tokens.jws_private_key_repository, 'private.pem')
    os.remove(private_key)
    self.assertRaises(SystemExit, jws.Provider)