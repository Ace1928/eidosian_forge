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
def test_empty_public_key_repo_raises_system_exception(self):
    for f in os.listdir(CONF.jwt_tokens.jws_public_key_repository):
        path = os.path.join(CONF.jwt_tokens.jws_public_key_repository, f)
        os.remove(path)
    self.assertRaises(SystemExit, jws.Provider)